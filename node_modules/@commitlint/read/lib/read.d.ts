interface GetCommitMessageOptions {
    cwd?: string;
    from?: string;
    to?: string;
    last?: boolean;
    edit?: boolean | string;
    gitLogArgs?: string;
}
export default function getCommitMessages(settings: GetCommitMessageOptions): Promise<string[]>;
export {};
//# sourceMappingURL=read.d.ts.map