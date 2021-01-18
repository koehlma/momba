(define (domain triangle-tire)
  (:requirements :typing :strips :equality :probabilistic-effects)
  (:functions (total-cost))
  (:types location)
  (:predicates (vehicle-at ?loc - location)
	       (spare-in ?loc - location)
	       (road ?from - location ?to - location)
	       (not-flattire) (hasspare))
  (:action move-car
    :parameters (?from - location ?to - location)
    :precondition (and (vehicle-at ?from) (road ?from ?to) (not-flattire))
    :effect (and (increase (total-cost) 1) (vehicle-at ?to) (not (vehicle-at ?from))
		 (probabilistic 0.5 (not (not-flattire)))))
  (:action loadtire
    :parameters (?loc - location)
    :precondition (and (vehicle-at ?loc) (spare-in ?loc))
    :effect (and (increase (total-cost) 1) (hasspare) (not (spare-in ?loc))))
  (:action changetire
    :precondition (hasspare)
    :effect (and (increase (total-cost) 1) (not (hasspare)) (not-flattire)))

)
