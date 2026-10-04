"""Print the 38-registration contract inventory without constructing environments."""

import json

from tests.helpers.reference.inventory import inventory

if __name__ == "__main__":
    print(json.dumps(inventory(), indent=2))
