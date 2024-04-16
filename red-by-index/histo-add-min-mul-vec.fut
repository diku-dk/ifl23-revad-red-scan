-- ==
--  entry: primalAdd revadAdd primalMin revadMin primalMul revadMul
-- compiled input@data/histo-31-10mil.in
-- compiled input@data/histo-401-10mil.in
-- compiled input@data/histo-500K-10mil.in
-- compiled input@data/histo-31-100mil.in
-- compiled input@data/histo-401-100mil.in
-- compiled input@data/histo-500K-100mil.in

def fadd [n][m] (is: [n]i64) (dst: [m]f32, vs: [n]f32) =
  reduce_by_index (copy dst) (+) 0 is vs

entry primalAdd [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (_ones: [m]f32) =
  fadd is (dst, vs)

entry revadAdd [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (ones: [m]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fadd is) (dst,vs) ones
  in  (r, adj_dst, adj_vs)

def fmin [n][m] (is: [n]i64) (dst: [m]f32, vs: [n]f32) =
  reduce_by_index (copy dst) (f32.min) f32.inf is vs

entry primalMin [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (_ones: [m]f32) =
  fmin is (dst, vs)

entry revadMin [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (ones: [m]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fmin is) (dst,vs) ones
  in  (r, adj_dst, adj_vs)

def fmul [n][m] (is: [n]i64) (dst: [m]f32, vs: [n]f32) =
  reduce_by_index (copy dst) (*) 1 is vs

entry primalMul [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (_ones: [m]f32) =
  fmul is (dst, vs)

entry revadMul [n][m] (is: [n]i64) (dst: [m]f32) (vs: [n]f32) (ones: [m]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fmul is) (dst,vs) ones
  in  (r, adj_dst, adj_vs)

-- ==
-- entry: primalVecAdd revadVecAdd primalVecMin revadVecMin primalVecMul revadVecMul
-- compiled input@data/histo-31-1milx10.in
-- compiled input@data/histo-401-1milx10.in
-- compiled input@data/histo-50K-1milx10.in
-- compiled input@data/histo-31-10milx10.in
-- compiled input@data/histo-401-10milx10.in
-- compiled input@data/histo-50K-10milx10.in

def fvec_add [n][m][d] (is: [n]i64) (dst: [m][d]f32, vs: [n][d]f32) =
  reduce_by_index (copy dst) (map2 (f32.+)) (replicate d f32.inf) is vs

entry primalVecAdd [n][m][d] (is: [n]i64) (dst: [m][d]f32) (vs: [n][d]f32) (_ones: [m][d]f32) =
  fvec_add is (dst, vs)

entry revadVecAdd [n][m][d] (is: [n]i64) (dst: [m][d]f32) (vs: [n][d]f32) (ones: [m][d]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fvec_add is) (dst,vs) ones
  in  (r, adj_dst, adj_vs)

--------------------------------------------------

def fvec_min [n][m][d] (is: [n]i64) (dst: [m][d]f32, vs: [n][d]f32) =
  reduce_by_index (copy dst) (map2 (f32.min)) (replicate d f32.inf) is vs

entry primalVecMin [n][m][d] (is: [n]i64) (dst: [m][d]f32) (vs: [n][d]f32) (_ones: [m][d]f32) =
  fvec_min is (dst, vs)

entry revadVecMin [n][m][d] (is: [n]i64) (dst: [m][d]f32) (vs: [n][d]f32) (ones: [m][d]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fvec_min is) (dst,vs) ones
  in  (r, adj_dst, adj_vs)

-------------------------------------------------

def fvec_mul [n][m][d] (is: [n]i64) (dst: [m][d]f32, vs: [n][d]f32) =
  reduce_by_index (copy dst) (map2 (*)) (replicate d 1) is vs

entry primalVecMul [n][m][d] (is: [n]i64) (dst: [m][d]f32) (vs: [n][d]f32) (_ones: [m][d]f32) =
  fvec_mul is (dst, vs)

entry revadVecMul [n][m][d] (is: [n]i64) (dst: [m][d]f32) (vs: [n][d]f32) (ones: [m][d]f32) =
  let (r, (adj_dst, adj_vs)) = vjp2 (fvec_mul is) (dst,vs) ones
  in  (r, adj_dst, adj_vs)
--  vjp (fvec_mul is) (dst,vs) (replicate m (replicate d 1))
