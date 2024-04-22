import { minLength } from '@commitlint/ensure';
export const headerMinLength = (parsed, _when = undefined, value = 0) => {
    var _a;
    return [
        minLength(parsed.header, value),
        `header must not be shorter than ${value} characters, current length is ${(_a = parsed.header) === null || _a === void 0 ? void 0 : _a.length}`,
    ];
};
//# sourceMappingURL=header-min-length.js.map