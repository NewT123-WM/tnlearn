import minimist from 'minimist';
import { getHistoryCommits } from './get-history-commits.js';
import { getEditCommit } from './get-edit-commit.js';
import { execa } from 'execa';
// Get commit messages
export default async function getCommitMessages(settings) {
    const { cwd, from, to, last, edit, gitLogArgs } = settings;
    if (edit) {
        return getEditCommit(cwd, edit);
    }
    if (last) {
        const gitCommandResult = await execa('git', [
            'log',
            '-1',
            '--pretty=format:%B',
        ]);
        let output = gitCommandResult.stdout;
        // strip output of extra quotation marks ("")
        if (output[0] == '"' && output[output.length - 1] == '"')
            output = output.slice(1, -1);
        return [output];
    }
    let gitOptions = { from, to };
    if (gitLogArgs) {
        gitOptions = {
            ...minimist(gitLogArgs.split(' ')),
            from,
            to,
        };
    }
    return getHistoryCommits(gitOptions, { cwd });
}
//# sourceMappingURL=read.js.map