# Momba: Model Representation (MombaIR)

A [JANI](https://jani-spec.org/)-inspired _intermediate representation_ (IR) for formal models.

**Rationale:** Directly ingesting JANI models in Rust with Serde via the usual derive macros is difficult, if not impossible, because the JANI specification makes use of arbitrary JSON schema constructs which do not map well to Serde's derive macros. Hence, this crate specifies an alternative JSON representation which is close to JANI but can be parsed more easily.

This crate uses [Sidex](https://oss.silitics.com/sidex/) for the specification of the structure and format of MombaIR.
