#!/bin/bash

chainlit run app.py -h -c &
sleep 10
npx cypress run --record false --config-file cypress/cypress.config.ts
kill %%
