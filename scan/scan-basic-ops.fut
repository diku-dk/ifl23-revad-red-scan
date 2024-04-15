import "scan-adj-comp"

-- Scan with scalar addition: performance
-- ==
-- entry: scan_add_prim scan_add_ours scan_add_comp
-- compiled random input { [10000000]f32   [10000000]f32 }
-- compiled random input { [100000000]f32  [100000000]f32 }

entry scan_add_prim [n] (inp : [n]f32) (_adj: [n]f32) : [n]f32 =
    scan (+) 1f32 inp

entry scan_add_comp [n] (inp : [n]f32) (adj: [n]f32) : [n]f32 =
    scan_bar 0f32 (+) (+) 1f32 inp adj

entry scan_add_ours [n] (inp : [n]f32) (adj: [n]f32) : [n]f32 =
    vjp (scan (+) 1f32) inp adj

-- Scan with vectorised addition: performance
-- ==
-- entry: scan_vecadd_prim scan_vecadd_ours scan_vecadd_comp
-- compiled random input { [16][1000000]f32  [16][1000000]f32 }
-- compiled random input { [16][10000000]f32 [16][10000000]f32 }

let zero m : [m]f32 = replicate m 0
let plus [m] (as: [m]f32) (bs: [m]f32) = map2 (+) as bs

def vecadd [k] (as: [k]f32) (bs: [k]f32) =
  map2 (+) as bs

def vecadd_ne k =
  replicate k 0f32

def primal_vecadd [n][k] (a: [n][k]f32) =
  scan vecadd (vecadd_ne k) (map opaque a)

entry scan_vecadd_prim [m][n] (inp : [m][n]f32) 
                              (_adj: [m][n]f32)
                            : [m][n]f32 =
  primal_vecadd (transpose inp)
  |> transpose

entry scan_vecadd_comp [m][n] (inp : [m][n]f32)
                              (adj : [m][n]f32)
                            : [m][n]f32 =
  scan_bar (zero m) plus vecadd (vecadd_ne m)
           (map opaque (transpose inp))
           (map opaque (transpose adj))
  |> transpose

entry scan_vecadd_ours [m][n] (inp : [m][n]f32)
                              (adj : [m][n]f32)
                            :  [m][n]f32 =
  let adj = vjp primal_vecadd (transpose inp) (transpose adj)
  in  transpose adj


-- Scan with scalar min: performance
-- ==
-- entry: scan_min_prim scan_min_ours scan_min_comp
-- compiled random input { [10000000]f32   [10000000]f32 }
-- compiled random input { [100000000]f32  [100000000]f32 }

entry scan_min_prim [n] (inp : [n]f32) (_adj: [n]f32) : [n]f32 =
    scan (f32.min) f32.highest inp

entry scan_min_comp [n] (inp : [n]f32) (adj: [n]f32) : [n]f32 =
    scan_bar 0f32 (+) (f32.min) f32.highest inp adj

entry scan_min_ours [n] (inp : [n]f32) (adj: [n]f32) : [n]f32 =
    vjp (scan (+) 1f32) inp adj


-- Scan with scalar multiplication: performance
-- ==
-- entry: scan_mul_prim scan_mul_ours scan_mul_comp 
-- compiled random input { [10000000]f32  [10000000]f32 }
-- compiled random input { [100000000]f32 [100000000]f32 }

entry scan_mul_prim [n] (inp : [n]f32) (_adj: [n]f32) : [n]f32 =
    scan (*) 1f32 inp

entry scan_mul_comp [n] (inp : [n]f32) (adj: [n]f32) : [n]f32 =
    scan_bar 0f32 (+) (*) 1f32 inp adj

entry scan_mul_ours [n] (inp : [n]f32) (adj: [n]f32) : [n]f32 =
    vjp (scan (*) 1f32) inp adj
    

-- Scan with vectorised multiplication: performance
-- ==
-- entry: scan_vecmul_prim scan_vecmul_ours scan_vecmul_comp 
-- compiled random input { [16][1000000]f32  [16][1000000]f32 }
-- compiled random input { [16][10000000]f32 [16][10000000]f32 }

def vecmul [k] (as: [k]f32) (bs: [k]f32) =
  map2 (*) as bs

def vecmul_ne k =
  replicate k 1f32

def primal_vecmul [n][k] (a: [n][k]f32) =
  scan vecmul (vecmul_ne k) (map opaque a)

entry scan_vecmul_prim [m][n] (inp : [m][n]f32) 
                              (_adj: [m][n]f32)
                            : [m][n]f32 =
  primal_vecmul (transpose inp)
  |> transpose

entry scan_vecmul_comp [m][n] (inp : [m][n]f32)
                              (adj : [m][n]f32)
                            : [m][n]f32 =
  scan_bar (zero m) plus vecmul (vecmul_ne m)
           (map opaque (transpose inp))
           (map opaque (transpose adj))
  |> transpose

entry scan_vecmul_ours [m][n] (inp : [m][n]f32)
                              (adj : [m][n]f32)
                            : [m][n]f32 =
  vjp primal_vecmul (transpose inp) (transpose adj)
  |> transpose


-- Scan with vectorised multiplication: validity
-- ==
-- entry: valid_vecmul
-- compiled random input {  [16][150000]f32  [16][150000]f32 } output {true}

let equalEps [n] (eps: f32) (as: [n]f32) (bs: [n]f32) : bool =
  map2 (\ a b -> (f32.abs (a-b)) <= eps ) as bs
  |> reduce_comm (&&) true

entry valid_vecmul [m][n] (inp : [m][n]f32)
                          (adj : [m][n]f32)
                        : bool =
  let comp = scan_bar (zero m) plus vecmul (vecmul_ne m)
                (map opaque (transpose inp))
                (map opaque (transpose adj))
          |> transpose

  let ours = vjp primal_vecmul (transpose inp) (transpose adj)
          |> transpose
  
  in  map2 (equalEps 0.00001f32) comp ours |> reduce (&&) true        
--  in  map2 (map2 (==)) comp ours
--      |> map (reduce (&&) true)
--      |> reduce (&&) true
