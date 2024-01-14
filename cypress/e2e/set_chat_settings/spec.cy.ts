describe('set_chat_settings', () => {
  before(() => {
    cy.visit('/')
  })

  it('should update inputs', () => {
    // Open chat settings modal
    cy.get('#chat-settings-open-modal').should('exist')
    cy.get('#chat-settings-open-modal').click()
    cy.get('#chat-settings-dialog').should('exist')

    // Update inputs
    const sliderInput = '{upArrow}{upArrow}{upArrow}{upArrow}{upArrow}'

    cy.get('#num_sources').clear().type('5')
    cy.get('#num_sources').should('have.value', '5')

    cy.get('#temperature').type(sliderInput)
    cy.get('#temperature').should('have.value', '0.9')

    cy.get('#repeat_penalty').type(sliderInput)
    cy.get('#repeat_penalty').should('have.value', '1.8')

    cy.get('#top_k').type(sliderInput)
    cy.get('#top_k').should('have.value', '25')

    cy.get('#top_p').type(sliderInput)
    cy.get('#top_p').should('have.value', '0.40')

    cy.contains('Confirm').click()

    cy.get('.step').should('have.length', 1)

    // Check if inputs are updated
    cy.get('#chat-settings-open-modal').click()
    cy.get('#num_sources').should('have.value', '5')
    cy.get('#temperature').should('have.value', '0.5')
    cy.get('#repeat_penalty').should('have.value', '0.9')
    cy.get('#top_k').should('have.value', '5')
    cy.get('#top_p').should('have.value', '0.5')

    // Check if modal is correctly closed
    cy.contains('Cancel').click()
    cy.get('#chat-settings-dialog').should('not.exist')
  })
})
