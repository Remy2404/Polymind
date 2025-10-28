---
applyTo: '**/*'
---

## Running 
- Use `uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload` to start dev server.
- Use ` uv run app.py` for prod server
 

## skip write a documnentation all context for this project 
when you write a code pls just write a code, do not write documentation for the whole project context.
## Testing 
- Save in dir tests/

# Please read this Instructions
# Copilot Instructions for Senior Software Engineers

This document defines best practices and rules for using AI coding assistants (e.g., Copilot) in a professional software engineering environment. The goal is to ensure **clean, secure, maintainable code** that aligns with project standards, architecture, and workflows.

---

## General Principles

* **Read the Codebase First**: Before generating, modifying, or refactoring code, the assistant must understand the existing structure, dependencies, and conventions.
* **Follow Project Structure**: Adhere strictly to the established architecture, coding conventions, and style guidelines.
* **Secure & Clean Code**: Write code with security, performance, and maintainability in mind.
* **Task Awareness**: Track and remember all tasks across sessions to maintain context continuity.

---

## Testing Requirements

* If new code requires testing:

  * Place **unit tests** or **integration tests** in the `tests/` folder.
  * Use the existing testing framework unless otherwise specified.
  * Ensure **100% reproducibility** of test results.

---


**Rules for MCP Tools:**

* Use the **right tool for the right task** automatically.
* Avoid unnecessary tool calls for trivial actions.
* Reference **tool documentation** before execution.
* Optimize for **efficiency** and **clarity** in output.

---

## Software Engineering Workflow

1. **Collaborative Design**: Align implementation with team agreements before coding.
2. **Code Reviews**: Ensure AI-generated code passes internal review before merging.
3. **Style Guidelines**: Maintain a **consistent coding style** across the entire codebase.

---

## OOP Principles

Follow core Object-Oriented Programming concepts for scalable and maintainable software:

1. **Encapsulation** – Keep internal states private; expose controlled interfaces.
2. **Inheritance** – Promote code reuse through class hierarchies.
3. **Polymorphism** – Design interchangeable components for flexibility.
4. **Abstraction** – Use interfaces/abstract classes for simplified design contracts.

---

## Avoid Anti-Patterns

The AI must avoid:

* **Tight Coupling** – Use abstractions/interfaces instead of hard dependencies.
* **Reinventing the Wheel** – Prefer reliable libraries over custom code for common tasks.
* **Ignoring Error Handling** – Always handle exceptions gracefully.
* **Overengineering** – Keep solutions simple, modular, and easy to maintain.
* **Hardcoding Values** – Use configuration files or environment variables for settings.
* **Ignoring Performance** – Optimize for performance, especially in critical code paths.
* **Neglecting Documentation** – Document complex logic and public APIs.
* **Skipping Tests** – Always write tests for new features or bug fixes.
* **Ignoring Security** – Follow the best security practices to protect data and systems.
* **Poor Naming Conventions** – Use clear, descriptive names for variables, functions, and classes.
* **Lack of Comments** – Comment non-obvious code to aid future maintainers.
* **Inconsistent Formatting** – Adhere to the project's coding style and formatting rules.
* **Code Duplication** – Avoid duplicating code; instead, create reusable components.
* **Ignoring Scalability** – Design systems that can grow with increased load or complexity.
---

## Security Guidelines

* **Data Privacy**: Never expose sensitive data in logs or responses.
* **API Security**: Always implement authentication/authorization controls.
* **Input Validation**: Sanitize and validate all inputs to prevent attacks.

---

## Clean Code Practices

* **Meaningful Names**: Use descriptive names for variables, methods, and classes.
* **Single Responsibility Principle**: Each class/function does one thing well.
* **Consistent Formatting**: Follow the project’s linter and formatter rules.
* **Code Comments**: Add comments for non-obvious logic; avoid redundant remarks.

---

## AI Control Directives

* Always **read the existing code** before suggesting changes.
* Maintain **context awareness** across related tasks.
* Generate **clean, secure, testable code** aligned with team conventions.
* **Proactively refactor** when the codebase can be improved without breaking functionality.
* Never introduce **hardcoded secrets** or **unapproved libraries**.