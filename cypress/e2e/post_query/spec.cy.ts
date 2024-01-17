describe('POST /query', () => {
  before(() => {
    cy.visit('/')
    cy.wait(30000)
    cy.get('#chat-input').type('How many eyestalks does a beholder have?{enter}')
    cy.get('.playground-button').should('exist').click()
  })
  it('test prompt', () => {
    cy.request({
      method: 'POST',
      url: 'http://localhost:8000/query',
      body: {
        'prompt': 'How many eyestalks does a Beholder have?'
      },
      failOnStatusCode: false
    }).then((response) => {
      expect(response.body.answer).length.to.be.greaterThan(1)
      expect(response.body.source_documents).length.to.be.greaterThan(1)
    })
  })
})
