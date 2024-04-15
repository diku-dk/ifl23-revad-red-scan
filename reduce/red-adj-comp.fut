def scan_right [n] 't (op : t -> t -> t) (e : t) (u : [n]t) : [n]t =
  reverse u |>
  scan (\ a b -> op b a) e |>
  reverse

def op_bar_1 't (op : t -> t -> t) (x: t, y: t, r_b: t) : t =
  let op' b a = op a b
  in  vjp (op' y) x r_b

def op_bar_2 't (op : t -> t -> t) (x: t, y: t, r_b: t) : t =
  vjp (op x) y r_b

def op_lft 't (plus: t -> t -> t)
              (op : t -> t -> t)
              (x1: t, a1: t, y1_h: t)
              (_x2: t, a2: t, y2_h: t) :
              (t, t, t) =
  let z_term = op_bar_1 op (x1, a1, y2_h)
  let z = plus z_term y1_h
  in  (x1, op a1 a2, z)

def reduce_bar [n] 't (zero: t)
                    (plus: t -> t -> t)
                    (op :  t -> t -> t)
                    (e : t) 
                    (u : [n]t)
                    (y_b : t) : [n]t =
  let x_b = map (\i -> if i == n-1 then y_b else zero) (iota n)
  let x = scan op e u
  let u_lft = map (\i -> if i < n-1 then u[i+1] else e) (iota n)
  let m = zip3 x u_lft x_b
  let (_, _, x_hat) = unzip3 <|
        scan_right (op_lft plus op) (e, e, zero) m
  let x_rht = map (\i -> if i == 0 then e else x[i-1]) (iota n)
  let u_bar = map (op_bar_2 op) (zip3 x_rht u x_hat)
  in  u_bar
