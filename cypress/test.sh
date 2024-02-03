#!/bin/bash

export TEST=true
chainlit run -h app.py &
sleep 10
npx cypress run --record false --config-file cypress/cypress.config.ts
kill %%
