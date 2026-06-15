Pipeline
========

Data flows from the LACE tracker's raw outputs to cross-animal statistics
in four stages, each independently re-runnable with file intermediates in
between.

.. code-block:: text

   raw files (.mat / .seq / .avi / .smr)
            │
            ▼
   ┌─────────────────────────────┐
   │ data_handlers               │  read MATLAB / video / Spike2
   └─────────────────────────────┘
            │
            ▼
   ┌─────────────────────────────┐
   │ trace_analysis              │  correct → mm trajectories,
   │  traceCorrector             │  curvature, speed, spikes
   │  traceAnalyser              │
   │  SpikeDetector / ...        │
   └─────────────────────────────┘
            │  result frames (CSV)
            ▼
   ┌─────────────────────────────┐
   │ fish_data_base              │  sort folder → analyse each →
   │  counterCurrentAna (sort)   │  per-fish CSV database
   │  fishRecAnalysis (per rec.) │
   │  fishDataBase (master CSV)  │
   └─────────────────────────────┘
            │  fishDataBase.csv
            ▼
   ┌─────────────────────────────┐
   │ data_base_analyser          │  cross-animal statistics
   │  CounterCurrentAnalyser     │
   │ plotting                    │  publication figures
   └─────────────────────────────┘

Stage 1 — ingest (``data_handlers``)
------------------------------------

``MatlabResultLoader`` unpacks a LACE ``*_result_ana.mat`` file into
trace, contour, mid-line, head/tail, and kinematic arrays.
``MediaHandler`` gives uniform frame access across OpenCV movies, Norpix
sequences, and image stacks. ``Spike2SimpleReader`` / ``SegmentSaver``
read electrophysiology into per-segment pandas frames.

Stage 2 — per-recording analysis (``trace_analysis``)
-----------------------------------------------------

``TraceCorrector`` aligns detection to the video (interactively if
needed). ``TraceAnalyser`` converts pixel traces to millimetres and
derives spatial histograms, in-zone metrics, and uniform mid-lines.
``CurvatureAnalyser``, ``SpeedAnalyser``, and ``SpikeDetector`` add
curvature, swimming kinematics, and spike trains.

Stage 3 — database assembly (``fish_data_base``)
------------------------------------------------

``SortMultiFileFolder`` groups a folder of raw files by genotype, sex,
and animal number. ``FishRecAnalysis`` runs one recording end-to-end and
writes its result frames (built by ``result_frames``) plus a database
row. ``FishDataBase`` grows and persists the master CSV.

Stage 4 — cross-animal statistics and figures
----------------------------------------------

``CounterCurrentAnalyser`` reads the occupancy densities back out of the
database and compares groups by distance from the stream centre. The
``plotting`` modules render the publication figures.
