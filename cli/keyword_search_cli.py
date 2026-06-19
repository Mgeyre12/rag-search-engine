#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, search_command, tokenize_term, InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Term Frequency")
    tf_parser.add_argument("docId", type=int, help="Document Id")
    tf_parser.add_argument("term", type=str, help="Search Term")


    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "tf":
            print("Looking up Term Frequency")
            term = tokenize_term(args.term)
            idx = InvertedIndex()
            idx.load()
            print(idx.get_tf(args.docId,term))


        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query,5)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
