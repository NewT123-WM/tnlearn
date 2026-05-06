# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" Class for evaluating programs proposed by the Sampler."""
from __future__ import annotations

from abc import abstractmethod, ABC
import ast
import time
from collections.abc import Sequence
import copy
from typing import Any, Type
import profile
import multiprocessing

from . import code_manipulation
from . import buffer
from . import evaluator_accelerate
from . import evaluate_on_problems

class _FunctionLineVisitor(ast.NodeVisitor):
    """ Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None: 
        """ Collect the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """ Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None 
        return self._function_end_line


def _trim_function_body(generated_code: str) -> str:
    """ Extract the body of the generated function, trimming anything after it.
    Please note that the indentation is REQUIRED !!!
    """
    if not generated_code:
        return ''

    code = f'def fake_function_header():\n{generated_code}'

    tree = None
    while tree is None:
        try:
            tree = ast.parse(code)
        
        except SyntaxError as e:
            if e.lineno is None: # Nothing could be saved when syntaxError
                return ''
            code = '\n'.join(code.splitlines()[:e.lineno - 1])

    if not code:
        return ''

    visitor = _FunctionLineVisitor('fake_function_header')
    visitor.visit(tree)
    body_lines = code.splitlines()[1:visitor.function_end_line]
    return '\n'.join(body_lines) + '\n\n'


def _sample_to_program(
        generated_code: str,
        version_generated: int | None,
        template: code_manipulation.Program,
        function_to_evolve: str,
) -> tuple[code_manipulation.Function, str]:
    """ 
    Return the compiled generated function and the full runnable program.
    This function removes the content after the generated function body.
    """
    body = _trim_function_body(generated_code)
    if version_generated is not None:
        body = code_manipulation.rename_function_calls(
            code=body,
            source_name=f'{function_to_evolve}_v{version_generated}',
            target_name=function_to_evolve
        )

    program = copy.deepcopy(template)
    evolved_function = program.get_function(function_to_evolve)
    evolved_function.body = body
    
    return evolved_function, str(program)


class Sandbox(ABC):
    """ Sandbox for executing generated code. """

    @abstractmethod
    def run(
            self,
            program: str,
            function_to_run: str,
            function_to_evolve: str,
            inputs: Any,  
            test_input: str, 
            timeout_seconds: int,
            **kwargs

    # ) -> tuple[Any, bool]:

    # 02 版本 输出报错信息
    ) -> tuple[Any, bool, str]:
        
        """ Return `function_to_run(test_input)` and whether execution succeeded. """
        raise NotImplementedError(
            'Must provide a sandbox for executing untrusted code.')


class LocalSandbox(Sandbox):
    """
    Secure environment for executing and evaluating LLM generated programs.
    Prevents harmful operations, limits resource usage, and enforces timeouts.
    Returns a 'score' for the executed program.
    """

    def __init__(self, verbose=False, numba_accelerate=False):
        """
        Initialize Sandbox.
        
        Args:
        verbose (bool): Enable detailed output.
        numba_accelerate (bool): Use Numba for acceleration of evaluation (limited compatibility). 
        """
        self._verbose = verbose
        self._numba_accelerate = numba_accelerate
        self._last_params = None

#################################### 02版本
    def run(self, program: str, function_to_run: str, function_to_evolve: str, 
        # 02 版本
        inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool, str]:
        # 原版
        # inputs: Any, test_input: str, timeout_seconds: int, **kwargs) -> tuple[Any, bool]:
        """
        Execute the given program sample and return its score and success status.
        
        Note: This sandbox is specific to the equation program skeleton discovery problem.
        """

        dataset = inputs[test_input]
        result_queue = multiprocessing.Queue() 
        process = multiprocessing.Process(
            target=self._compile_and_run_function,
            args=(program, function_to_run, function_to_evolve, dataset, self._numba_accelerate, result_queue)
        )
        process.start()
        process.join(timeout=timeout_seconds)

        # if the process is not finished in time, terminate
        if process.is_alive():
            process.terminate()
            process.join()
            # results = None, False
            # 02 版本
            results = None, None,False, 'timeout01'
        else:
            results = self._get_results(result_queue)
        
        if self._verbose:
            self._print_evaluation_details(program, results, **kwargs)
        # print(len(results)) #4
        # 这里对results进行处理,先拆开成四份，再把grade和runs_ok放到一起
        if results and len(results) in (4,5):
            if len(results) == 5:
                grade, res, runs_ok, remark, params = results
                self._last_params = params
            else:
                grade, res, runs_ok, remark = results
                self._last_params = None
            results = (grade, runs_ok,remark)
            # print(f"拆分后的grade: {grade}, res: {res}, runs_ok: {runs_ok}, remark: {remark}")
        else:
            res = None
            # print('*********************')
            # print(self._print_evaluation_details(program, results, **kwargs))
        # print("result:------------")
        # print(results)

        return results,res


    def _get_results(self, queue):
        #临时修改为1方便查看输出
        for _ in range(1):
            if not queue.empty():
                return queue.get_nowait()
            time.sleep(0.1)
        # return None, False
        # 02 版本
        return None, False, 'timeout02'


    def _print_evaluation_details(self, program, results, **kwargs):
        print('================= Evaluated Program =================')
        function = code_manipulation.text_to_program(program).get_function(kwargs.get('func_to_evolve', 'equation'))
        print(f'{str(function).strip()}\n-----------------------------------------------------')
        print(f'Score: {results}\n=====================================================\n\n')



    def _compile_and_run_function(self, program, function_to_run, function_to_evolve, 
                                  dataset, numba_accelerate, result_queue):
        try:
            # optimize the code (decorate function_to_run with @numba.jit())
            if numba_accelerate:
                program = evaluator_accelerate.add_numba_decorator(
                    program=program,
                    function_to_evolve=function_to_evolve
                )
            
            # execute the program, map func/var/class to global namespace
            all_globals_namespace = {}
            exec(program, all_globals_namespace)
            function_to_run = all_globals_namespace[function_to_run]
            evolved_function = all_globals_namespace[function_to_evolve]
            eval_out = evaluate_on_problems.evaluate(dataset, evolved_function, verbose=self._verbose)
            # 兼容两种返回格式：旧(分数, 矩阵) / 新(分数, 矩阵, 参数)
            if isinstance(eval_out, tuple) and len(eval_out) == 3:
                results, full_res, opt_params = eval_out
            else:
                results, full_res = eval_out
                opt_params = None
            if full_res is not None and hasattr(full_res, "shape") and len(full_res) > 0:
                import numpy as np
                # 确定采样数量，最多取20个点或全部（如果数据量小于20）
                sample_size = min(100, len(full_res))
                # 随机选择索引，不排序，保持随机性
                indices = np.random.choice(len(full_res), sample_size, replace=False)
                # 对残差进行采样
                res = full_res[indices]
                # print(f"残差随机采样（{sample_size}个点）:", res)
            if not isinstance(results, (int, float)):
                result_queue.put((None, False, 'no output'))
                return
            
            ########################### 是不是这里加一段‘yes’就可以了？
            result_queue.put((results, res, True, 'yes', opt_params))
            
        # if raise any exception, execution is failed
        except Exception as e:
            # print(f"Execution Error: {e}")
            # result_queue.put((None, False))

            # 把报错信息输出
            error_msg = f"Execution Error: {e}"
            print('eeeeeeerrrrrrrrrroooooorrrrrrrr')
            print(error_msg)
            result_queue.put((None, None, False, error_msg, None))



def _calls_ancestor(program: str, function_to_evolve: str) -> bool:
    """ Return whether the generated function is calling an earlier version. """
    for name in code_manipulation.get_functions_called(program):
        if name.startswith(f'{function_to_evolve}_v'):
            return True
    return False



class Evaluator:
    """ Class that analyses functions generated by LLMs. """

    def __init__(
            self,
            database: buffer.ExperienceBuffer,
            template: code_manipulation.Program,
            function_to_evolve: str, 
            function_to_run: str, 
            inputs: Sequence[Any], 
            timeout_seconds: int = 30,
            sandbox_class: Type[Sandbox] = Sandbox
    ):
        self._database = database
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._function_to_run = function_to_run
        self._inputs = inputs
        self._timeout_seconds = timeout_seconds
        self._sandbox = sandbox_class()

    def analyse(
            self,
            sample: str,
            island_id: int | None,
            version_generated: int | None,
            **kwargs 
    # ) -> None:
    
    # ) -> float:
    ) -> tuple[float, str]:
        
        # tuple[Any, bool, str]
        """ Compile the hypothesis sample into a program and executes it on test inputs. """
        new_function, program = _sample_to_program(
            sample, version_generated, self._template, self._function_to_evolve)
        scores_per_test = {}

        time_reset = time.time()

        # print('len of self._inputs: ',len(self._inputs))    # len of self._inputs:  1
        # print(self._inputs) # x1 x2
        '''
        {'data': {'inputs': array([[-0.25197899, -0.17306601],
       [-0.25232508, -0.17300887],
       [-0.25267104, -0.17295167],
       ...,
       [-0.41992701,  0.11309208],
       [-0.41970063,  0.11328232],
       [-0.41947387,  0.11347256]]), 'outputs': array([0.0285521 , 0.02858525, 0.02861839, ..., 0.09512003, 0.09511695,
       0.09511373])}}
        '''
        
        # print('len of self._inputs: ',len(self._inputs))    # len of self._inputs:  1
        # print(bbbbb)
        for current_input in self._inputs:
            
            # test_output, runs_ok = self._sandbox.run(


            # 02 版本 收集错误信息
            # test_output, runs_ok, error_msg = self._sandbox.run(
            results, res= self._sandbox.run(
                program, self._function_to_run, self._function_to_evolve, self._inputs, current_input,
                self._timeout_seconds
            )
            test_output, runs_ok, error_msg = results
            if runs_ok and not _calls_ancestor(program, self._function_to_evolve) and test_output is not None:
                if not isinstance(test_output, (int, float)):
                    print(f'Error: test_output is {test_output}')
                    raise ValueError('@function.run did not return an int/float score.')
                scores_per_test[current_input] = test_output

        evaluate_time = time.time() - time_reset
        ###################
        # print("error_msg=========")
        # print(error_msg)
        # print(test_output)      # score: -0.0004185108785400066 为针对初始化方程框架的评分
        # print('我从analyse中拿到了res', res) 
        # print(bbb)


        # 果代码运行成功并得到有效评分，分数会被保存到经验缓冲区(ExperienceBuffer)：
        '''
        这里的_database就是从sampler.py传入的buffer.ExperienceBuffer实例。它将：

        将函数与其评分一起保存
        将函数分配到适当的"岛屿"(island)中
        根据功能相似性将函数组织到集群(clusters)中
        '''
        if scores_per_test:
            # 将优化参数保存到函数对象，便于 Profiler 写入 samples JSON
            try:
                params = getattr(self._sandbox, '_last_params', None)
                new_function.optimized_params = params
            except Exception:
                pass

            self._database.register_program(
                new_function,
                island_id,
                scores_per_test,
                **kwargs,
                evaluate_time=evaluate_time
            )
        
        else:
            profiler: profile.Profiler = kwargs.get('profiler', None)
            if profiler:
                global_sample_nums = kwargs.get('global_sample_nums', None)
                sample_time = kwargs.get('sample_time', None)
                new_function.global_sample_nums = global_sample_nums
                new_function.score = None
                new_function.sample_time = sample_time
                new_function.evaluate_time = evaluate_time
                try:
                    params = getattr(self._sandbox, '_last_params', None)
                    new_function.optimized_params = params
                except Exception:
                    pass
                profiler.register_function(new_function)
        
        
        return test_output, error_msg, res
