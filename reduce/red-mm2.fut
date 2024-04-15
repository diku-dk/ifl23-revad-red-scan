import "red-adj-comp"

-------------------------------------------------
---  3. Scan with 2x2 Matrix Multipplication  ---
-------------------------------------------------

def mm2by2  (a1: f32, b1: f32, c1: f32, d1: f32)
            (a2: f32, b2: f32, c2: f32, d2: f32) =
  ( a1*a2 + b1*c2
  , a1*b2 + b1*d2
  , c1*a2 + d1*c2
  , c1*b2 + d1*d2
  )

def mm2_ne = (1f32, 0f32, 0f32, 1f32)

def primal [n] (xs: [n](f32,f32,f32,f32)) =
  reduce mm2by2 mm2_ne xs

def fromarrs = map (\(x: [4]f32) -> (x[0],x[1],x[2],x[3]))
def toarrs = map (\(a,b,c,d) -> [a,b,c,d])


let zero = (0f32, 0f32, 0f32, 0f32)
let plus (a1: f32, a2: f32, a3: f32, a4: f32)
         (b1: f32, b2: f32, b3: f32, b4: f32) =
  (a1+b1, a2+b2, a3+b3, a4+b4)


-- Scan with 2x2 matrix multiplication: correctness
-- ==
-- entry: red_mm2_prim red_mm2_ours red_mm2_comp
-- compiled random input { [10000000]f32 [10000000]f32 [10000000]f32 [10000000]f32 f32 f32 f32 f32 }
-- compiled random input { [100000000]f32 [100000000]f32 [100000000]f32 [100000000]f32 f32 f32 f32 f32 }

entry red_mm2_prim [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (inp4 : [n]f32)
                        (_adj1 : f32)
                        (_adj2 : f32) 
                        (_adj3 : f32)
                        (_adj4 : f32) : 
                        (f32,f32,f32,f32) =
  zip4 inp1 inp2 inp3 inp4 |> primal

entry red_mm2_comp [n] (inp1 : [n]f32)
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (inp4 : [n]f32)
                        (adj1 : f32)
                        (adj2 : f32)
                        (adj3 : f32)
                        (adj4 : f32) : 
                        ([n]f32,[n]f32,[n]f32,[n]f32) =
  reduce_bar zero plus mm2by2 mm2_ne
            (zip4 inp1 inp2 inp3 inp4)
            (adj1, adj2, adj3, adj4) 
    |> unzip4

entry red_mm2_ours [n] (inp1 : [n]f32) 
                        (inp2 : [n]f32)
                        (inp3 : [n]f32)
                        (inp4 : [n]f32)
                        (adj1 : f32)
                        (adj2 : f32)
                        (adj3 : f32)
                        (adj4 : f32) : 
                        ([n]f32,[n]f32,[n]f32,[n]f32) =
  vjp primal (zip4 inp1 inp2 inp3 inp4) (adj1, adj2, adj3, adj4)
    |> unzip4
