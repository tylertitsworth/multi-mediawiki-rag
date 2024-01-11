import { execSync } from 'child_process';

export const ROOT = process.cwd();
export const CHAINLIT_PORT = 8000;

export function runTestServer(env?: Record<string, string>) {
    cy.exec(`npx ts-node cypress/run.ts`, {env});
    cy.visit('/');
}


export function runCommand(command: string, cwd = ROOT) {
    return execSync(command, {
        encoding: 'utf-8',
        cwd,
        env: process.env,
        stdio: 'inherit'
    });
}

export async function runTests() {
    // Cypress requires a healthcheck on the server at startup so let's run
    // Chainlit before running tests to pass the healthcheck
    runCommand('npx ts-node cypress/run.ts');

    // Recording the cypress run is time consuming. Disabled by default.
    // const recordOptions = ` --record --key ${process.env.CYPRESS_RECORD_KEY} `;
    return runCommand(
        `npx cypress run --record false --config-file cypress/cypress.config.ts`
    );
}
