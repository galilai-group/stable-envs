"""Compatibility CLI for the shared native acceptance witnesses."""

from tests.helpers.reference.native_acceptance import main

if __name__ == "__main__":
    main(("cartpole", "pendulum"))
