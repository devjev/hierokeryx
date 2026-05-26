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
        - resolve_incremental
        - build_registry

## `hierokeryx.resolve.centroids`

::: hierokeryx.resolve.centroids
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - RegistryCentroids
        - compute_centroids
        - update_centroids
        - save_centroids
        - load_centroids
        - centroids_paths

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
