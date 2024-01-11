import { runTests } from './utils';

async function main() {
    await runTests();
}

main()
    .then(() => {
        console.log('Done!');
        process.exit(0);
    })
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });
