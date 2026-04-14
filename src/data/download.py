"""
Download do dataset Digipathos (Embrapa).

TODO: implementar o download real após confirmar a URL/API final do Digipathos.
      URL base: https://www.digipathos-rep.cnptia.embrapa.br
"""

import argparse
from pathlib import Path


def download_digipathos(target: Path) -> None:
    """
    Baixa o dataset Digipathos para `target`.

    TODO: implementar quando a URL de download em massa for confirmada.
          Por enquanto, faça o download manual e coloque as imagens em:
          target/<nome_da_classe>/*.jpg
    """
    raise NotImplementedError(
        "Download automático do Digipathos ainda não implementado.\n"
        "Baixe manualmente em https://www.digipathos-rep.cnptia.embrapa.br "
        f"e extraia para: {target}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baixa dataset Digipathos")
    parser.add_argument("--target", type=Path, required=True, help="Diretório de destino")
    args = parser.parse_args()
    download_digipathos(args.target)
