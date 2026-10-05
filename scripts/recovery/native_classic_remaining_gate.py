"""Compatibility CLI for the shared native acceptance witnesses."""

from tests.helpers.reference.native_acceptance import main

if __name__ == "__main__":
    main(("acrobot", "mountain_car", "mountain_car_continuous"))
