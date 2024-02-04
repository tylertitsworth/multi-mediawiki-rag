describe('set_chat_settings', () => {
  before(() => {
    cy.visit('/')
  })

  it('should update inputs', () => {
    // Open chat settings modal
    cy.get('#chat-settings-open-modal').should('exist')
    cy.get('#chat-settings-open-modal').click()
    cy.get('#chat-settings-dialog').should('exist')

    cy.get('#num_sources').clear().type('5')
    cy.get('#num_sources').should('have.value', '5')

    cy.get('#temperature').clear().type('0.7')
    cy.get('#temperature').should('have.value', '0.7')

    cy.get('#repeat_penalty').type('{upArrow}{upArrow}').trigger('change')
    cy.get('#repeat_penalty').should('have.value', '2')

    cy.get('#top_k').type('{upArrow}{upArrow}').trigger('change')
    cy.get('#top_k').should('have.value', '22')

    cy.get('#top_p').clear().type('0.77')
    cy.get('#top_p').should('have.value', '0.77')

    cy.contains('Confirm').click()

    cy.get('.step').should('have.length', 1)

    // Check if inputs are updated
    cy.get('#chat-settings-open-modal').click()
    cy.get('#num_sources').should('have.value', '5')
    cy.get('#temperature').should('have.value', '0.7')
    cy.get('#repeat_penalty').should('have.value', '2')
    cy.get('#top_k').should('have.value', '22')
    cy.get('#top_p').should('have.value', '0.77')

    // Check if modal is correctly closed
    cy.contains('Cancel').click()
    cy.get('#chat-settings-dialog').should('not.exist')
  })
})
