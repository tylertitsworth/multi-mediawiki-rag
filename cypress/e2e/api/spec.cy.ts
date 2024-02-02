describe('test api', () => {
  before(() => {
    cy.visit('/')
    cy.wait(30000)
    cy.get('#chat-input').type('What is the Armor Class of a Beholder?{enter}')
    cy.get('.playground-button').should('exist').click()
  })
  it('GET /ping', () => {
    cy.request({
      method: 'GET',
      url: 'http://localhost:8000/ping',
      failOnStatusCode: false
    }).then((response) => {
      expect(response.body.status).equal('Healthy')
    })
  })
  it('POST /query', () => {
    cy.request({
      method: 'POST',
      url: 'http://localhost:8000/query',
      body: {
        prompt: 'What is the Armor Class of a Beholder?'
      },
      failOnStatusCode: false
    }).then((response) => {
      expect(response.body.answer).length.to.be.greaterThan(1)
      expect(response.body.source_documents).length.to.be.greaterThan(1)
    })
  })
})
