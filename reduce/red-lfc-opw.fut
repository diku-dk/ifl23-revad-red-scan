import "red-adj-comp"

-- Reduce with linear-function composition and sum-of-products operators: performance
-- ==
-- entry: red_lfc_comp red_lfc_ours red_lfc_prim red_opw_comp red_opw_ours red_opw_optm red_opw_prim 
-- compiled random input { [10000000]f32  [10000000]f32 f32 f32 }
-- compiled random input {[100000000]f32 [100000000]f32 f32 f32 }

def plus_tup (a1: f32, a2: f32) (b1: f32, b2: f32) = (a1+b1, a2+b2)
def zero_tup = (0f32, 0f32)

-- linear-function composition (lfc) operator;
let lfc (a0: f32, a1: f32)
        (b0: f32, b1: f32) : (f32, f32) =
    (b0 + b1*a0, a1*b1)

-- neutral element for linear-function composition
let lfc_ne = (0f32, 1f32)

def primal_lfc [n] (xs: [n](f32,f32)) =
  reduce lfc lfc_ne xs

entry red_lfc_prim [n] (inp1 : [n]f32) 
                       (inp2 : [n]f32)
                       (_adj1 : f32)
                       (_adj2 : f32) : 
                       (f32,f32) =
  zip inp1 inp2 |> primal_lfc

entry red_lfc_comp [n] (inp1 : [n]f32)
                       (inp2 : [n]f32)
                       (adj1 : f32)
                       (adj2 : f32) : 
                       ([n]f32,[n]f32) =
  reduce_bar zero_tup plus_tup lfc lfc_ne (zip inp1 inp2) (adj1, adj2) 
    |> unzip

entry red_lfc_ours [n] (inp1 : [n]f32) 
                       (inp2 : [n]f32)
                        (adj1 : f32)
                        (adj2 : f32) = 
                 --     : ((f32,f32), [n](f32,f32)) =
  let adj_zip = vjp primal_lfc (zip inp1 inp2) (adj1, adj2)
  in unzip adj_zip
    
--------------
--- Reduce with fency commutative and associative operator:
--- reduce op [w1, w2, w3] = w1*w2 + w1*w3 + w2*w3
--- ToDo:
---  simplify it so that it is mapped on an arrays of weights
---    and only then reduced
--------------

def opw (p1: f32, s1: f32) (p2: f32, s2: f32) = (p1 + p2 + s1*s2, s1+s2)
def opw_inv (acc_p: f32, acc_s: f32) (p1: f32, s1: f32) = 
  let s2 = acc_s - s1
  let p2 = acc_p - p1 - s1*s2
  in  (p2, s2)

def opw_ne = (0f32, 0f32)
def opw_lft (w: f32) = (0f32, w)

def primal_opw [n] (xs: [n](f32,f32)) =
  reduce_comm opw opw_ne xs

let vjp_opw_optm [n] (inp1 : [n]f32) 
                      (inp2 : [n]f32)
                      (adj1 : f32)
                      (adj2 : f32) : 
                      ([n]f32, [n]f32) =
  let (r1, r2) = reduce_comm opw opw_ne (zip inp1 inp2)
  let (inp1_, inp2_) = unzip <|
    map2 (\ p1 s1 ->
            let (p2, s2) = opw_inv (r1,r2) (p1, s1)
            let (p1_bar, s1_bar) = vjp (opw (p2, s2)) (p1, s1) (adj1, adj2)
            in  (p1_bar, s1_bar)
         ) inp1 inp2
  in (inp1_, inp2_)


entry red_opw_prim [n] (inp1 : [n]f32) 
                       (inp2 : [n]f32)
                       (_adj1 : f32)
                       (_adj2 : f32) : 
                       (f32,f32) =
  zip inp1 inp2 |> primal_opw

entry red_opw_comp [n] (inp1 : [n]f32)
                       (inp2 : [n]f32)
                       (adj1 : f32)
                       (adj2 : f32) : 
                       ([n]f32,[n]f32) =
  reduce_bar zero_tup plus_tup opw opw_ne (zip inp1 inp2) (adj1, adj2) 
    |> unzip

entry red_opw_ours [n] (inp1 : [n]f32) 
                       (inp2 : [n]f32)
                       (adj1 : f32)
                       (adj2 : f32)
                     : ([n]f32,[n]f32) =
  let adj_zip =
    vjp primal_opw (zip inp1 inp2) (adj1, adj2)
  in unzip adj_zip
    
entry red_opw_optm [n] (inp1 : [n]f32) 
                       (inp2 : [n]f32)
                       (adj1 : f32)
                       (adj2 : f32) : 
                       ([n]f32,[n]f32) =
  vjp_opw_optm inp1 inp2 adj1 adj2


-- Reduce with linear-function composition: validity
-- ==
-- entry: valid_opw
-- compiled random input {  [1500]f32  [1500]f32 f32 f32 } output {true}

let equalEps [n] (eps: f32) (as: [n]f32) (bs: [n]f32) : bool =
  map2 (\ a b -> (f32.abs (a-b)) <= eps ) as bs
  |> reduce_comm (&&) true

entry valid_opw [n] (inp1 : [n]f32) 
                    (inp2 : [n]f32)
                    (adj1 : f32)
                    (adj2 : f32) =
 let (comp_bar1, comp_bar2) =
    reduce_bar zero_tup plus_tup opw opw_ne (zip inp1 inp2) (adj1, adj2) 
    |> unzip
 let (ours_bar1, ours_bar2) =
    vjp primal_opw (zip inp1 inp2) (adj1, adj2)
    |> unzip
 let (optm_bar1, optm_bar2) =
    vjp_opw_optm inp1 inp2 adj1 adj2
 
 let eps = 0.0002f32
 in  equalEps eps comp_bar1 ours_bar1 &&
     equalEps eps ours_bar1 optm_bar1 &&
     equalEps eps comp_bar2 ours_bar2 &&
     equalEps eps ours_bar2 optm_bar2

