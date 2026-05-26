"""Incremental cross-doc resolution: batch, then add new docs against it.

Demonstrates `hkx pipeline ... --against <existing-workdir>` from Python:
the first run produces a workdir with registry + centroid sidecar; the
second run resolves new documents against those existing clusters,
joining them where the entity matches and creating new clusters
otherwise.

Run inside the project's nix dev shell:

    nix develop
    export GATEWAY_BASE_URL=... GATEWAY_API_KEY=... GATEWAY_API_VERSION=...
    uv run python examples/incremental_pipeline.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hierokeryx import Document, EntitySchema, EntityType, pipeline
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.llm.standard_gateway_client import StandardGatewayClient
from hierokeryx.resolve.embed import SentenceTransformerEmbedder


def main() -> None:
    schema = EntitySchema(
        types=[
            EntityType(name="Person", description="A named individual human"),
            EntityType(name="Organization", description="A company, agency, institution"),
            EntityType(name="Location", description="A geographic place"),
        ]
    )

    batch1 = [
        Document(
            id="curie-long",
            text=(
                "Marie Curie was a Polish-French physicist. Curie discovered radium "
                "with her husband Pierre in Paris. She won two Nobel Prizes."
            ),
        ),
    ]
    batch2 = [
        Document(
            id="curie-short",
            text=(
                "M. Curie discovered radium with her husband Pierre. The couple "
                "worked at the Sorbonne in Paris."
            ),
        ),
        Document(
            id="einstein",
            text=(
                "Albert Einstein published the theory of relativity in 1915. "
                "Einstein later moved to Princeton."
            ),
        ),
    ]

    workdir_root = Path("examples-workdir")
    if workdir_root.exists():
        shutil.rmtree(workdir_root)
    wd1 = workdir_root / "batch1"
    wd2 = workdir_root / "batch2"

    extractor = GLiNERExtractor(model_id="urchade/gliner_small-v2.1")
    llm = StandardGatewayClient()
    embedder = SentenceTransformerEmbedder()

    # First batch — produces wd1/ with registry + centroid sidecar.
    run1 = pipeline.run(
        documents=batch1,
        schema=schema,
        workdir=wd1,
        extractor=extractor,
        llm_client=llm,
        embedder=embedder,
    )
    print(f"[batch1] {len(run1.registry.clusters)} cluster(s) → {wd1}")
    for cid, members in run1.registry.clusters.items():
        canonical = run1.registry.canonical_by_cluster[cid]
        print(f"  {cid}: {canonical} ({len(members)} member(s))")

    # Second batch — incremental against batch1.
    run2 = pipeline.run(
        documents=batch2,
        schema=schema,
        workdir=wd2,
        extractor=extractor,
        llm_client=llm,
        embedder=embedder,
        incremental_from=wd1,
    )
    print(f"\n[batch2 incremental] {len(run2.registry.clusters)} cluster(s) → {wd2}")
    for cid, members in run2.registry.clusters.items():
        canonical = run2.registry.canonical_by_cluster[cid]
        marker = "+" if cid not in run1.registry.clusters else " "
        print(f"  {marker} {cid}: {canonical} ({len(members)} member(s))")

    print(
        "\nClusters prefixed `+` are new; the others were extended by entities "
        "from batch2."
    )


if __name__ == "__main__":
    main()
