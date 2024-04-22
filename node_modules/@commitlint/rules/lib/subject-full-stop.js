import message from '@commitlint/message';
export const subjectFullStop = (parsed, when = 'always', value = '.') => {
    var _a;
    const colonIndex = ((_a = parsed.header) === null || _a === void 0 ? void 0 : _a.indexOf(':')) || 0;
    if (colonIndex > 0 && colonIndex === parsed.header.length - 1) {
        return [true];
    }
    const input = parsed.header;
    const negated = when === 'never';
    let hasStop = (input === null || input === void 0 ? void 0 : input[input.length - 1]) === value;
    if ((input === null || input === void 0 ? void 0 : input.slice(-3)) === '...') {
        hasStop = false;
    }
    return [
        negated ? !hasStop : hasStop,
        message(['subject', negated ? 'may not' : 'must', 'end with full stop']),
    ];
};
//# sourceMappingURL=subject-full-stop.js.map