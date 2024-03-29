describe('PromptPlayground', () => {
  beforeEach(() => {
    cy.visit('/')
    cy.wait(30000)
    cy.get('#chat-input').type('What is the Armor Class of a Beholder?{enter}')
    cy.get('.playground-button').should('exist').click()
  })
  it('template variables', () => {
    cy.get('#playground').should('exist')
  })
  it('template prompt', () => {
    cy.get('#submit-prompt').should('exist')
    cy.get('.completion-editor [contenteditable]').should('not.be.empty')
  })
  it('chat settings', () => {
    cy.get('#temperature').invoke('val').should('equal', '0')
    cy.get('#repeat_penalty').invoke('val').should('equal', '1.8')
    cy.get('#top_k').invoke('val').should('equal', '20')
    cy.get('#top_p').invoke('val').should('equal', '0.35')
  })
})
