describe('PromptPlayground', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.wait(30000)
    cy.get('#chat-input').type("How many eyestalks does a beholder have?{enter}")
    cy.get('.playground-button').should('exist').click()
  })
  it('template variables', () => {
    cy.get('.input-question').should('exist')
    cy.get('.input-context').should('exist')
  })
  it('template prompt', () => {
    cy.get('#submit-prompt').should('exist')
    cy.get('.completion-editor [contenteditable]').should(
      'contain',
      "ten eyestalks"
    )
  })
  it('chat settings', () => {
    cy.get('#temperature').invoke('val').should('equal', '0.3')
    cy.get('#repeat_penalty').invoke('val').should('equal', '1.6')
    cy.get('#top_k').invoke('val').should('equal', '20')
    cy.get('#top_p').invoke('val').should('equal', '0.35')
  })
})
