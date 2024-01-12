#!/bin/bash

chainlit run main.py -h -c &
sleep 10
npx cypress run --record false --config-file cypress/cypress.config.ts
kill %%
