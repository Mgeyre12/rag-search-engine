#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="verify model")


    embed_parser = subparsers.add_parser("embed_text", help="embed text")
    embed_parser.add_argument("query", type=str, help="embedding query")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="verify embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="verify embeddings")
    embed_query_parser.add_argument("query", type=str, help="embedding query")  

    search_parser = subparsers.add_parser("search", help="Search movies using semantic serch")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="limit parameter")

    embed_query_parser.add_argument("query", type=str, help="embedding query")  

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_command(args.query,args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()