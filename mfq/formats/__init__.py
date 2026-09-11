"""MFQ native-format definitions.

- ``nint`` -- One version-2 neuron-anchored INT container; uniform bit-widths
  are quantizer presets rather than public tensor dtypes.
- ``nvq`` -- One NVQ container whose payload selects the E8/D4/JSC or
  ultra-low-bit profile.
- ``npq0_l`` / ``npq0_s`` -- Internal profile codecs of the public NPQ
  container.
- ``nepq`` -- One NEPQ container for cross-expert shared-table profiles.
- ``mxfp4_sq`` -- One self-describing MXFP4-SQ container for SQ2/SQ3.
- ``mfe`` -- Mixed Format Experts; nested cohort dtypes are family-level.
- ``ternary`` -- Neuron-anchored scalar ternary reference format with five trits per byte.
- ``scheme`` -- Precision schemes describing weight/activation precision for every tensor (for example MFQ-W4.51).
- ``header`` -- MFQ file headers and tensor-level metadata.
- ``io`` -- MFQ file serialization and deserialization.
"""
