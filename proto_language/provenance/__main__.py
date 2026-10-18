"""Allow ``python -m proto_language.provenance``; see :mod:`proto_language.provenance.cli`."""

import sys

from proto_language.provenance.cli import main

if __name__ == "__main__":
    sys.exit(main())
