# `hierokeryx.resolve`

Within-document coreference and cross-document entity resolution.

## `hierokeryx.resolve.coref`

::: hierokeryx.resolve.coref
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - resolve_within_doc

## `hierokeryx.resolve.crossdoc`

::: hierokeryx.resolve.crossdoc
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - resolve_crossdoc
        - build_registry

## `hierokeryx.resolve.cluster`

::: hierokeryx.resolve.cluster
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - ClusterAssignment
        - cluster_by_type
        - apply_assignments

## `hierokeryx.resolve.embed`

::: hierokeryx.resolve.embed
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - SentenceTransformerEmbedder
        - build_entity_repr
