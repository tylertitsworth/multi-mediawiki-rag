import { defineConfig } from 'cypress'

export default defineConfig({
  projectId: 'iqos6r',
  component: {
    devServer: {
      framework: 'react',
      bundler: 'vite'
    }
  },
  viewportWidth: 1200,

  e2e: {
    supportFile: false,
    defaultCommandTimeout: 100000,
    video: false,
    baseUrl: 'http://127.0.0.1:8000',
    setupNodeEvents (on) {
      on('task', {
        log (message) {
          console.log(message)
          return null
        }
      })
    }
  }
})
