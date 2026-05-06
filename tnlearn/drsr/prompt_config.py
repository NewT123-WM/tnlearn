"""
集中管理 DRSR 的 LLM 提示词模板。

这里不再保留旧的 oscillator 专用默认文案，统一对齐到当前共享 spec：
- 外层 prompt 使用 x0/x1/.../y 变量命名；
- 若有 metadata，则在 prompt 中带上物理语义；
- 避免继续向模型暴露 with driving force / col0 / col1 之类的历史模板残留。
"""

# 任务头中使用的占位参数（用于 _do_request 中的 head 文本格式化）
problem_name_in_prompt = 'target relation'
dependent_name_in_prompt = 'y'
independent_name_in_prompt = 'x0 and x1'


# 采样阶段：说明性指令（拼接在代码 prompt 前）
instruction_prompt = (
    "You are a helpful assistant tasked with discovering mathematical function structures for scientific systems. "
    "Complete the 'equation' function below, considering the physical meaning and relationships of inputs.\n\n"
)

# 采样后经验对话整体模板（包含上下文与追问占位）
analysis_conversation_template = (
    "Here's our previous conversation:\n\n"
    "user: {prompt}\n\n"
    "assistant: {sample}\n\n"
    "user: {question}\n"
)


# 采样后“基于得分的经验总结”三种追问模板
analysis_question_good = (
    "The optimized function skeleton you just answered scored higher. Please summarize useful experience.\n"
    "STRICTLY follow these rules:\n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent} in {problem}, I can ...\"\n"
    "2. Summarize ONLY the key success factors\n"
    "3. You need to make your answer as concise as possible\n"
)

analysis_question_bad = (
    "The optimized function skeleton you just answered scored lower. What lessons can you draw from it?\n"
    "STRICTLY follow these rules: \n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent} in {problem}, I can ...\"\n"
    "2. Identify ONE crucial improvement point\n"
    "3. You need to make your answer as concise as possible\n"
)

analysis_question_none = (
    "The optimized function skeleton you just answered failed with error: {error}. What lessons can you draw from it?\n"
    "{budget_sentence}"
    "STRICTLY follow these rules:\n"
    "1. Use the exact phrasing \"when seeking for the mathematical function skeleton that represents {dependent} in {problem}, I need ...\"\n"
    "2. Address the SPECIFIC error: {error}\n"
    "3. Treat this failed sample as a negative example to avoid, not as a target requirement to satisfy\n"
    "4. Identify ONE concrete change that would prevent the next sample from repeating this failure\n"
    "5. You need to make your answer as concise as possible\n"
)

# 经验注入区块标题与条目前缀
ideas_block_title = "\n\n### The following are ideas summarized based on past experiences in solving such problems. ###\n\n"
idea_item_prefix = "idea{index}：\n"

# 残差分析注入区块标题
residual_block_title = ("\n\n### The following is the analysis result of the existing data on {problem}, "
                        "which will assist you in answering the question. ###\n\n")


# 采样阶段：任务头（追加在发送前）
head_template = (
    "Find the mathematical function skeleton that represents {dependent} in {problem}, "
    "given data on {independent}. \n"
)


# 残差分析提示模板（包含固定格式与输出要求）
residual_analysis_prompt = (
    "You are a data analysis expert.\n"
    "previous conclusions:{last_analysis}\n"
    "dataset:{residual}\n"
    "The equation corresponding to the residuals:{sample}\n\n"
    "The independent variables are x0 and x1.\n"
    "The dependent variable is y.\n"
    "The forth column contains residuals (calculated as observed value - predicted value from the equation).\n"
    "Each row represents a set of independent variables and the corresponding dependent variable and residual.\n\n"
    "Task Requirements:\n\n"
    "1. Please analyze and summarize the influence of the changes in the values of different independent variables on the dependent variable,\n"
    "as well as the possible intrinsic relationships among different independent variables.\n\n"
    "Your response only needs to answer your analysis results in the form below, and you don't need to show your analysis process.\n\n"
    "2.##Output Format##:\n"
    "STRICTLY deliver results in the following structured format:\n\n"
    "  \"output_format\": {\n"
    "    \"analysis\": {\n"
    "      \"independent_to_dependent_relationships\": {\n"
    "        \"x0 \": [\n"
    "          \"Hint: analyze the functional relationship between x0 and y in different intervals\"\n"
    "        ],\n"
    "        \"x1 \": [\n"
    "          \"Hint: analyze the functional relationship between x1 and y in different intervals\"\n"
    "        ]\n"
    "      },\n"
    "      \"inter_relationships_between_independents\": {\n"
    "        \"x0 vs x1\": [\n"
    "          \"Hint: analyze the possible functional relationship between x0 and x1 in different intervals. If not, leave blank.\"\n"
    "        ]\n"
    "      }\n"
    "    }\n"
    "  }\n"
)


# ==========================
# 动态渲染：无装饰器版本的上下文类
# ==========================

DEFAULT_BACKGROUND = "The physical properties of this equation are unknown and need to be analyzed based on experience."

def _ensure_feature_names(n, names):
    """确保有 n 个自变量名；缺省则按 x1..xN 生成。"""
    if names is None:
        return [f"x{i+1}" for i in range(n)]
    if len(names) != n:
        raise ValueError(f"feature_names 长度应为 {n}，实际为 {len(names)}")
    return names

def _ind_phrase(names):
    """生成 “x1, x2, and x3” 风格短语。"""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"

def _pairwise(names):
    """两两组合。"""
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j]))
    return pairs


class PromptContext:
    """提示词渲染上下文（无装饰器版本）。

    用法：
        ctx = PromptContext(n_features=X.shape[1], feature_names=None, dependent_name=None,
                            problem_name=None, background=None)
        head = ctx.render_head()
        instruction = ctx.render_instruction()
        q = ctx.render_analysis_question('Good')
        residual_prompt = ctx.render_residual_analysis_prompt(last_analysis, residual, sample)
    """

    def __init__(
        self,
        n_features,
        feature_names=None,
        dependent_name=None,
        problem_name=None,
        background=None,
        feature_descriptions=None,
        target_description=None,
        max_params=None,
    ):
        self.n_features = n_features
        self.feature_names = feature_names
        self.dependent_name = dependent_name
        self.problem_name = problem_name
        self.background = background
        self.feature_descriptions = feature_descriptions
        self.target_description = target_description
        self.max_params = max_params

    # 规范化后的属性
    @property
    def features(self):
        return _ensure_feature_names(self.n_features, self.feature_names)

    @property
    def dependent(self):
        return self.dependent_name or "y"

    @property
    def problem(self):
        return self.problem_name or problem_name_in_prompt

    @property
    def background_text(self):
        return (self.background or DEFAULT_BACKGROUND).strip()

    @property
    def normalized_feature_descriptions(self):
        features = self.features
        descriptions = self.feature_descriptions or []
        result = []
        for idx, name in enumerate(features):
            desc = descriptions[idx] if idx < len(descriptions) else None
            result.append((name, str(desc).strip() if desc and str(desc).strip() else None))
        return result

    @property
    def dependent_text(self):
        desc = self.target_description
        if desc and str(desc).strip():
            return f"{self.dependent} ({str(desc).strip()})"
        return self.dependent

    @property
    def max_param_count(self):
        try:
            value = int(self.max_params)
        except Exception:
            return None
        return value if value > 0 else None

    def _feature_phrase(self):
        items = []
        for name, desc in self.normalized_feature_descriptions:
            if desc:
                items.append(f"{name} ({desc})")
            else:
                items.append(name)
        return _ind_phrase(items)

    def _variables_block(self):
        lines = ["- Independents:"]
        for name, desc in self.normalized_feature_descriptions:
            if desc:
                lines.append(f"  - {name}: {desc}")
            else:
                lines.append(f"  - {name}")
        lines.append("- Dependent:")
        if self.target_description and str(self.target_description).strip():
            lines.append(f"  - {self.dependent}: {str(self.target_description).strip()}")
        else:
            lines.append(f"  - {self.dependent}")
        return "\n".join(lines)

    # 渲染方法
    def render_instruction(self):
        return (
            instruction_prompt
            + "Variables:\n"
            + f"{self._variables_block()}\n"
            + f"Background: {self.background_text}\n"
        )

    def render_head(self):
        return head_template.format(
            dependent=self.dependent_text,
            problem=self.problem,
            independent=self._feature_phrase(),
        )

    def render_analysis_question(self, quality, error=None):
        if quality == "Good":
            return analysis_question_good.format(dependent=self.dependent, problem=self.problem)
        if quality == "Bad":
            return analysis_question_bad.format(dependent=self.dependent, problem=self.problem)
        if quality == "None":
            max_params = self.max_param_count
            if max_params is None:
                budget_sentence = (
                    "Treat this failure as a negative example rather than a requirement to satisfy. "
                    "If the error is about parameter length or indexing, do not solve it by asking for more parameters. "
                    "Instead, reduce parameter usage so the equation fits the evaluator's available parameter budget.\n"
                )
            else:
                max_index = max_params - 1
                budget_sentence = (
                    f"The current evaluator passes exactly {max_params} trainable parameters, "
                    f"indexed from params[0] to params[{max_index}]. "
                    "Treat this failure as a negative example rather than a requirement to satisfy. "
                    "If the error is about parameter length or indexing, do not solve it by asking for more parameters. "
                    f"Instead, rewrite the equation so it stays within params[0]..params[{max_index}] "
                    f"and avoid any explicit minimum-length checks above {max_params}.\n"
                )
            return analysis_question_none.format(
                dependent=self.dependent,
                problem=self.problem,
                error=str(error or ""),
                budget_sentence=budget_sentence,
            )
        raise ValueError(f"unknown quality: {quality}")

    def render_residual_block_title(self):
        return residual_block_title.format(problem=self.problem)

    def render_residual_analysis_prompt(self, last_analysis, residual, sample):
        inds = self.features
        dep = self.dependent

        role_lines = [
            "The independent variables are:",
            *[
                f"- {name}: {desc}" if desc else f"- {name}"
                for name, desc in self.normalized_feature_descriptions
            ],
            "",
            f"The dependent variable is {self.dependent_text}.",
            "The forth column contains residuals (observed - predicted).",
        ]
        role_text = "\n".join(role_lines)

        ind_to_dep = "\n".join([
            f'        "{name} ": [\n'
            f'          "Hint: analyze the functional relationship between {name} and {dep} in different intervals"\n'
            f"        ],"
            for name in inds
        ])

        pairs = _pairwise(inds)
        inter_lines = "\n".join([
            f'        "{a} vs {b}": [\n'
            f'          "Hint: analyze possible functional relationship between {a} and {b} in different intervals. If not, leave blank."\n'
            f"        ]"
            for a, b in pairs
        ])
        if not inter_lines:
            inter_lines = '        "": []'

        dynamic_part = (
            f"{role_text}\n\n"
            "Task Requirements:\n\n"
            "1. Analyze and summarize how changes of each independent variable influence the dependent variable, "
            "and the possible intrinsic relationships among independent variables.\n\n"
            "Your response should follow the structure below; no need to show the reasoning process.\n\n"
            '2.##Output Format##:\n'
            'STRICTLY deliver results in the following structured format:\n\n'
            '  "output_format": {\n'
            '    "analysis": {\n'
            '      "independent_to_dependent_relationships": {\n'
            f"{ind_to_dep}\n"
            '      },\n'
            '      "inter_relationships_between_independents": {\n'
            f"{inter_lines}\n"
            '      }\n'
            '    }\n'
            '  }\n'
        )

        return (
            "You are a data analysis expert.\n"
            f"Background: {self.background_text}\n"
            f"previous conclusions:{last_analysis}\n"
            f"dataset:{residual}\n"
            f"The equation corresponding to the residuals:{sample}\n\n"
            f"{dynamic_part}"
        )
