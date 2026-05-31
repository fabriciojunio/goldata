"""CLI do GolData: comandos de linha de comando."""

import argparse
import sys
import json


def cmd_xg(args: argparse.Namespace) -> None:
    """Calcula xG para um chute."""
    from goldata.data.features import FeatureEngineer
    fe = FeatureEngineer()
    distance = fe.calculate_distance_to_goal(args.x, args.y)
    angle = fe.calculate_angle_to_goal(args.x, args.y)
    xg_approx = max(0.01, min(0.95, 0.3 - distance * 0.008 + angle * 0.15))
    if args.penalty:
        xg_approx = 0.76
    result = {"x": args.x, "y": args.y, "distance": round(distance, 2),
              "angle": round(angle, 4), "xg": round(xg_approx, 4)}
    print(json.dumps(result, indent=2))


def cmd_standings(args: argparse.Namespace) -> None:
    """Mostra classificação do Brasileirão."""
    from goldata.data.brasileirao import BrasileiraoDataClient
    client = BrasileiraoDataClient()
    df = client.get_serie_a_standings(args.season)
    print(f"\n{'#':>3} {'Time':<20} {'Pts':>4} {'J':>3} {'V':>3} {'E':>3} {'D':>3}")
    print("-" * 45)
    for _, row in df.head(20).iterrows():
        pos = int(row.get("position", 0))
        print(f"{pos:>3} {row['team']:<20} {int(row['points']):>4} {int(row.get('matches', 38)):>3} "
              f"{int(row.get('wins', 0)):>3} {int(row.get('draws', 0)):>3} {int(row.get('losses', 0)):>3}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Inicia a API FastAPI."""
    import uvicorn
    uvicorn.run(
        "goldata.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="goldata",
        description="GolData: Football Analytics Platform | Ictus Technologies",
    )
    sub = parser.add_subparsers(dest="command", help="Comando")

    # xg
    xg_parser = sub.add_parser("xg", help="Calcular xG de um chute")
    xg_parser.add_argument("--x", type=float, required=True, help="Posição x (0-120)")
    xg_parser.add_argument("--y", type=float, required=True, help="Posição y (0-80)")
    xg_parser.add_argument("--penalty", action="store_true", help="É pênalti?")

    # standings
    st_parser = sub.add_parser("standings", help="Classificação do Brasileirão")
    st_parser.add_argument("--season", type=int, default=2024, help="Temporada (padrão: 2024)")

    # serve
    srv_parser = sub.add_parser("serve", help="Iniciar API FastAPI")
    srv_parser.add_argument("--host", default="0.0.0.0")
    srv_parser.add_argument("--port", type=int, default=8000)
    srv_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "xg":
        cmd_xg(args)
    elif args.command == "standings":
        cmd_standings(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
