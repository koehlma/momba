;;;  Authors: Michael Littman and David Weissman  ;;;
;;;  Modified: Blai Bonet for IPC 2006 ;;;
;;;  Modified: Marcel Steinmetz ;;;

(define (domain tire)
  (:requirements :typing :strips :equality :probabilistic-effects)
  (:types location)
  (:predicates (vehicle-at ?loc - location) (spare-in ?loc - location) (road ?from - location ?to - location) (not-flattire) (hasspare))
  (:functions (total-cost))
  (:action move-car
    :parameters (?from - location ?to - location)
    :precondition (and (vehicle-at ?from) (road ?from ?to) (not-flattire))
    :effect (and (increase (total-cost) 1) (vehicle-at ?to) (not (vehicle-at ?from)) (probabilistic 2/5 (not (not-flattire))))
  )
  (:action loadtire
    :parameters (?loc - location)
    :precondition (and (vehicle-at ?loc) (spare-in ?loc))
    :effect (and (increase (total-cost) 1) (hasspare) (not (spare-in ?loc)))
  )
  (:action changetire
    :precondition (hasspare)
    :effect (and (increase (total-cost) 1) (probabilistic 1/2 (and (not (hasspare)) (not-flattire))))
  )
)

