import { spawn } from 'child_process';
import sh from 'shell-exec';

import { CHAINLIT_PORT, ROOT } from './utils';

interface CmdResult {
    stdout: string;
    stderr: string;
}

const killPort = async (port: number): Promise<CmdResult> => {
    return sh(`lsof -nPi :${port}`).then((res) => {
        const { stdout } = res;
        if (!stdout) return Promise.reject(`No process running on port ${port}`);
        return sh(
            `lsof -nPi :${port} | grep 'LISTEN' | awk '{print $2}' | xargs kill -9`
        );
    });
};

export const runChainlitForTest = async () => {
    try {
        await killPort(CHAINLIT_PORT);
        console.log(`Process on port ${CHAINLIT_PORT} killed`);
    } catch (error) {
        console.log(`Could not kill process on port ${CHAINLIT_PORT}. ${error}.`);
    }
    return new Promise((resolve, reject) => {
        // Headless + CI mode
        const options = [
            '-m',
            'chainlit',
            'run',
            'main.py',
            '-h',
            '-c'
        ];
        const server = spawn('python3', options, {
            cwd: ROOT
        });

        server.stdout.on('data', (data) => {
            console.log(`stdout: ${data}`);
            if (data.toString().includes('Your app is available at')) {
                resolve(server);
            }
        });

        server.stderr.on('data', (data) => {
            console.error(`stderr: ${data}`);
        });

        server.on('error', (error) => {
            reject(error.message);
        });

        server.on('exit', function (code) {
            reject('child process exited with code ' + code);
        });
    });
};

runChainlitForTest()
    .then(() => {
        process.exit(0);
    })
    .catch((error) => {
        console.error(error);
        process.exit(1);
    })
