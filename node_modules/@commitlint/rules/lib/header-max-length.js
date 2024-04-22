import { maxLength } from '@commitlint/ensure';
export const headerMaxLength = (parsed, _when = undefined, value = 0) => {
    var _a;
    return [
        maxLength(parsed.header, value),
        `header must not be longer than ${value} characters, current length is ${(_a = parsed.header) === null || _a === void 0 ? void 0 : _a.length}`,
    ];
};
//# sourceMappingURL=header-max-length.js.map