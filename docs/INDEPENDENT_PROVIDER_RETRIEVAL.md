# Independent provider semantic retrieval

`IndependentSemanticRetriever` encodes a query once per requested provider and
searches only that provider's matching collection. Its response contains a map
of provider ID to ranked list; it does not fuse or compare raw model scores.

Every result contains a 1-based contiguous rank, provider-local score,
`model_id`, `keyframe_id`, `video_id`, `frame_id`, `shot_id`, and `timestamp`.
Identity fields come from indexed canonical metadata, never filenames.

Two failure policies are available:

- `fail_closed`: an error in a configured required provider fails the request.
- `degraded`: successful provider lists are returned with explicit coverage and
  per-provider errors. A failed provider is never replaced by another model.

Inner-product/cosine indexes sort larger scores first; distance indexes sort
smaller values first. Ties use `keyframe_id`, and duplicate hits retain the
provider-local best score. These rules apply only within one provider list.
