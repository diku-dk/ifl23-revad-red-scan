import "red-adj-comp"

-- Reduce with scalar addition: performance
-- ==
-- entry: red_add_prim red_add_ours red_add_comp 
-- compiled random input { [10000000]f32 f32 }
-- compiled random input {[100000000]f32 f32 }

def primal_add [n] (xs: [n]f32) : f32 = reduce (+) 0.0f32 xs

entry red_add_prim [n] (inp : [n]f32) (_adj1 : f32) : f32 =
  primal_add inp

entry red_add_comp [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  reduce_bar 0.0f32 (+) (+) 0.0f32 inp adj

entry red_add_ours [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  vjp primal_add inp adj


-- Reduce with scalar min: performance
-- ==
-- entry: red_min_prim red_min_ours red_min_comp 
-- compiled random input { [10000000]f32 f32 }
-- compiled random input {[100000000]f32 f32 }

def primal_min [n] (xs: [n]f32) : f32 = reduce (f32.min) f32.inf xs

entry red_min_prim [n] (inp : [n]f32) (_adj1 : f32) : f32 =
  primal_min inp

entry red_min_comp [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  reduce_bar 0.0f32 (+) (f32.min) f32.inf inp adj

entry red_min_ours [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  vjp primal_min inp adj


-- Reduce with scalar multiplication: performance
-- ==
-- entry: red_mul_prim red_mul_ours red_mul_comp red_mul_comp2
-- compiled random input { [10000000]f32 f32 }
-- compiled random input {[100000000]f32 f32 }

def primal_mul [n] (xs: [n]f32) : f32 = reduce (*) 1.0f32 xs

entry red_mul_prim [n] (inp : [n]f32) (_adj1 : f32) : f32 =
  primal_mul inp

entry red_mul_comp [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  let one = opaque 1.0f32
  let primal = reduce (\ a b -> a * one * b) 1.0f32
  in  vjp primal inp adj

entry red_mul_comp2 [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  reduce_bar 0.0f32 (+) (*) 1.0f32 inp adj

entry red_mul_ours [n] (inp : [n]f32) (adj : f32) : [n]f32 =
  vjp primal_mul inp adj


-- Reduce with vectorised multiplication: performance
-- ==
-- entry: red_vecmul_prim red_vecmul_comp red_vecmul_ours 
-- compiled random input { [16][1000000]f32  [16]f32 }
-- compiled random input { [16][10000000]f32 [16]f32 }

def vecmul [k] (as: [k]f32) (bs: [k]f32) =
  map2 (*) as bs

def vecmul_ne k =
  replicate k 1f32

def primal_vecmul [n][k] (a: [n][k]f32) =
  reduce_comm vecmul (vecmul_ne k) a

entry red_vecmul_prim [n][m] (inp : [m][n]f32) 
                             (_adj: [m]f32) : [m]f32 =
  primal_vecmul (transpose inp)

entry red_vecmul_comp [n][m] (inp : [m][n]f32)
                             (adj : [m]f32) : [m][n]f32 =
  let zero m : [m]f32 = replicate m 0
  let plus [m] (as: [m]f32) (bs: [m]f32) = map2 (+) as bs
  
  in reduce_bar (zero m) plus vecmul (vecmul_ne m)
                (map opaque (transpose inp))
                adj
   |> transpose

entry red_vecmul_ours [n][m] (inp : [m][n]f32)
                             (adj : [m]f32) : [m][n]f32 =
  let adj_inp = vjp primal_vecmul (transpose inp) adj
  in  transpose adj_inp

