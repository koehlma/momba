(define (problem ex_bw_5_p01)
  (:domain exploding-blocksworld)
  (:objects b1 b2 b3 b4 b5 - block)
  (:init (= (total-cost) 0)  (emptyhand) (on b1 b4) (on-table b2) (on b3 b2) (on b4 b5) (on-table b5) (clear b1) (clear b3) (no-detonated b1) (no-destroyed b1) (no-detonated b2) (no-destroyed b2) (no-detonated b3) (no-destroyed b3) (no-detonated b4) (no-destroyed b4) (no-detonated b5) (no-destroyed b5) (no-destroyed-table))
  (:goal (and  (on b2 b4) (on-table b4)  )
)
  
)
