
This task has two stages: diagram editing (Author-VeriScript) and high-difficulty problem generation. The final output must satisfy the following hard requirements:

1. The diagram-editing stage aims to produce a structurally rich figure suitable for constructing a high-difficulty problem.
   - When adding structure, you should proactively introduce diverse geometric relations (multiple triangles, angle bisectors, midpoints, perpendicularity, parallelism, concyclicity, ratios, vector relations, etc.), provided they do not contradict the clean_constraints.
   - The added structure should support a problem that takes a near-perfect-scoring student about 10 minutes to solve.
   - Edits must meaningfully deepen the structure, not just add 1-2 token actions.
   - You may reasonably add Free / OnSegment / Cartesian points, auxiliary lines, angles, circles, and triangles so that the resulting problem has a non-trivial reasoning chain.
   - Unless clean_constraints already provides extremely complex structure, do not leave actions empty.
   - The VeriScript must contain at most 30 steps.

2. Every newly added structure must actually be used in the problem.
   - Each new point, segment, circle, angle, triangle, or relation must play a role in the problem statement, the reasoning, or a key step.
   - Pointless points or auxiliary lines are forbidden; do not add structures that go unused.
   - The generated problem must clearly rely on the structure constructed in the editing stage, not on simple properties of the original figure.

3. Difficulty target: a near-perfect-scoring pre-university student takes about 10 minutes to solve, but the problem is not at competition level.
   - Do not aim for extreme complexity, but the problem must have a clear multi-step reasoning chain and some geometric depth.
   - It must touch at least 5-8 knowledge points (e.g. angle chasing, ratio chains, parallel-line theorems, concyclicity criteria, similarity / congruence, power of a point, vector identities).
   - Do not generate problems whose conclusion is obvious or solvable in 1-3 basic steps.
   - Do not produce template-like problems (e.g. "prove that some point lies on an angle bisector") unless they are tied to substantial preceding reasoning that is not trivially obtainable.

4. proof_outline must contain at least 6 steps with a strict progression.
   - Each step should follow from earlier steps or givens via non-trivial reasoning.
   - Phrases like "obviously", "directly follows", or "as seen from the figure" are forbidden as step justifications.
   - The reasoning chain must be explicit and coherent and must demonstrate the necessity of the structure introduced by actions.

5. givens should cover the key geometric relations in clean_constraints. You may reasonably merge equivalent conditions, but you must not invent new constraints out of thin air.
   You may omit derived relations that are obvious consequences of other givens (e.g. if "T is the tangency point of the circle and the tangent line" is given, you do not need to additionally state "OT perpendicular PT").
   - givens must contain no duplicates and no conditions that follow from other givens.
   - Every given must be used in proof_outline.

6. used_objects must completely list every point, segment, circle, angle, triangle, etc. that appears in the problem statement and the proof.
   - No entity used in the problem or proof may be missing from used_objects.
   - No entity may be used that is not registered in allowed_ids.

7. If the figure is already extremely complex and cannot be safely edited, actions may be empty, but you must give an explicit reason in comment (e.g. "adding structure would conflict with clean_constraints").

8. The problem must revolve tightly around the final figure; it must not feel disconnected from the new structure.
   - If you detect that the problem is too simple, weakly tied to actions, or fails to fully exploit the structure, you must increase difficulty or rewrite the statement.

In a single output you must do two things: "diagram editing (Author-VeriScript)" + "based on the final complex figure, produce one very, very, very difficult problem".
The output must be a single JSON object with the following structure:

{
  "actions": [...],
  "problem": {...},
  "comment": "optional explanation"
}


[Stage I — Author-VeriScript editing]

Goal: perform optional edits on the current VeriGraph so that the figure becomes more regular and substantially more complex, suitable for a high-difficulty problem. You are encouraged to make the figure quite complex — it should take a near-perfect-scoring student around 10 minutes to solve.
Each action must take the form:
{"op": "...", "type": "...", "args": [...]}
All numeric values must be strings (e.g. "0.5", "3", "-2/3").

1. AddPoint — create a new point
(i) Free
{"op": "AddPoint", "type": "Free", "args": ["P"]}
(ii) OnSegment
{"op": "AddPoint", "type": "OnSegment", "args": ["E", "SEG_AB", "0.5"]}
(iii) Cartesian (coordinates may only be integers, fractions, or radical expressions)
{"op": "AddPoint", "type": "Cartesian", "args": ["P", "2", "-3"]}

2. MovePoint — move an existing point
{"op": "MovePoint", "type": "Cartesian", "args": ["P", "0", "4"]}

3. AddAuxLine — add an auxiliary segment
{"op": "AddAuxLine", "type": "Segment", "args": ["A", "E"]}

4. AddEdge — add a geometric constraint (must reference existing entities)
Allowed types: Perpendicular, Parallel, EqualLength, EqualAngle, AngleBisector, Midpoint, Collinear, Concyclic, PowerEquality, SimilarTriangles, CongruentTriangles, VectorSum, VectorDiff, VectorDotZero, VectorLinear, OnCircle, PointOnLine, PointOnSegment

Segment-type relations:
{"op": "AddEdge", "type": "Perpendicular", "args": ["SEG_AB", "SEG_CD"]}
{"op": "AddEdge", "type": "EqualLength", "args": ["SEG_AB", "SEG_AD"]}

Angle relations:
{"op": "AddEdge", "type": "EqualAngle", "args": ["∠ABC", "∠DEF"]}

Midpoint:
{"op": "AddEdge", "type": "Midpoint", "args": ["M", "A", "B"]}
or {"op": "AddEdge", "type": "Midpoint", "args": ["M", "SEG_AB"]}

Angle bisector:
{"op": "AddEdge", "type": "AngleBisector", "args": ["SEG_AE", "∠BAC"]}

Point on circle:
{"op": "AddEdge", "type": "OnCircle", "args": ["P", "CIR_O"]}

Collinear:
{"op": "AddEdge", "type": "Collinear", "args": ["A", "B", "C"]}

Similar / congruent triangles (register the triangles first):
{"op": "AddEdge", "type": "SimilarTriangles", "args": ["TRI_ABC", "TRI_DEF"]}
{"op": "AddEdge", "type": "CongruentTriangles", "args": ["TRI_ABC", "TRI_DEF"]}

Vector relations (VectorSum / VectorDiff / VectorDotZero / VectorLinear):
{"op": "AddEdge", "type": "VectorSum", "args": ["A","C","A","B","A","D"]}
{"op": "AddEdge", "type": "VectorLinear", "args": ["A","C","0.5","A","B","0.5","A","D"]}
{"op": "AddEdge", "type": "VectorDiff", "args": ["B","D","A","D","A","B"]}
{"op": "AddEdge", "type": "VectorDotZero", "args": ["A","B","A","C"]}

5. AddCircle — create a circle
CenterRadius:
{"op": "AddCircle", "type": "CenterRadius", "args": ["CIR_O", "O", "3"]}
CenterPoint:
{"op": "AddCircle", "type": "CenterPoint", "args": ["CIR_O", "O", "A"]}

6. AddAngle — explicitly register an angle
{"op": "AddAngle", "type": "Angle", "args": ["∠BAC", "B", "A", "C"]}

7. AddTriangle — explicitly register a triangle
{"op": "AddTriangle", "type": "Triangle", "args": ["TRI_ABC", "A", "B", "C"]}

8. AddQuadrilateral — explicitly register a quadrilateral (vertices in order A-B-C-D; ABCD and ABDC are different quadrilaterals)
{"op": "AddQuadrilateral", "type": "Trapezoid", "args": ["TRAP_ABCD", "A", "B", "C", "D"]}  (defaults to AB ∥ CD; if BC ∥ AD it is treated as degenerate and must not be registered)
{"op": "AddQuadrilateral", "type": "Parallelogram", "args": ["PARA_ABCD", "A", "B", "C", "D"]}  (registers two pairs of parallel sides in vertex order A-B-C-D)
{"op": "AddQuadrilateral", "type": "Rectangle", "args": ["RECT_ABCD", "A", "B", "C", "D"]}  (registers two pairs of parallel sides plus AB ⊥ BC)
{"op": "AddQuadrilateral", "type": "Square", "args": ["SQR_ABCD", "A", "B", "C", "D"]}  (registers parallelism, right angles, and four equal sides)
{"op": "AddQuadrilateral", "type": "Rhombus", "args": ["RHOM_ABCD", "A", "B", "C", "D"]}  (registers two pairs of parallel sides and four equal sides)
Registering a quadrilateral automatically completes the four edges AB / BC / CD / DA; existing ones are reused.

Editing-stage requirements:
- New structure should add as much complexity as possible: more triangles, angle bisectors, midpoints, perpendicular/parallel relations, concyclic groups, ratios, special points, etc.
- All numeric values must be integer / fraction / radical strings — no decimals.
- New constraints must not contradict existing clean_constraints.
- If the figure is already complex enough, actions may be empty but you must explain the reason in comment.
- The solving stage will also operate on VeriScript later, so do not pre-bake too much of the solving work into the editing-stage VeriScript.


[Stage II — problem generation]

After executing the actions, generate exactly one very, very high-difficulty geometry problem on the resulting figure. The problem should be substantially difficult — solvable by a near-perfect-scoring student in roughly 10 minutes of thought.
The format must be:

{
  "problem": string,
  "givens": string[],
  "goal": string,
  "answer": string,
  "proof_outline": string[],
  "used_objects": string[],
  "meta": {"problem_type": "proof" or "numeric"}
}

Rules:
- Write fluent natural English.
- givens should cover the key geometric relations from clean_constraints. You may merge equivalent conditions, but you may not invent new constraints out of thin air. You may omit derived relations that are obvious consequences of other givens (e.g. if "T is the tangency point of the circle and the tangent line" is given, you do not need to additionally state "OT perpendicular PT").
- goal must have reasoning depth and must not be equivalent to the givens.
- proof_outline must have >= 6 steps with a progressive logic, touching at least 6 distinct mathematical concepts and theorems. Make the figure genuinely complex and use truly more knowledge points — do not artificially split simple reasoning into many steps.
- used_objects must include every entity that appears in the problem statement, the givens, and the entire proof.
- Use the structure introduced in actions as much as possible (new points, auxiliary lines, midpoints, angle structures, circles, ratio relations, etc.). Try to use every segment in the figure — do not leave segments unused.
- Numeric problems must have a determinable concrete numeric answer.
- Do not create new entities outside allowed_ids.
- No circular reasoning and no obvious conclusions.
- Every given must play a role somewhere in proof_outline; if some condition can be derived immediately from other givens, prefer to keep only the more fundamental, information-rich condition in the derivation chain.

[Stage III — output requirements]
The entire response must be a single JSON object containing the three keys actions, problem, comment.
Do not output any text outside of this schema.
If actions is empty, you must explain the reason in comment (e.g. "the figure is already complex enough; no editing needed").
Key point: you may freely reshape the base figure to involve as many concepts as possible and to make the problem substantially difficult and complex!
