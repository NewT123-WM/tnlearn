export class CliError extends Error {
    __proto__ = Error;
    type;
    error_code;
    constructor(message, type, error_code = 1) {
        super(message);
        this.type = type;
        this.error_code = error_code;
        Object.setPrototypeOf(this, CliError.prototype);
    }
}
//# sourceMappingURL=cli-error.js.map