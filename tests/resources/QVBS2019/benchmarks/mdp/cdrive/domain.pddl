(define (domain drive)
        (:requirements :typing :probabilistic-effects :equality)

        (:types coord direction color delay preference length rotation)

        (:predicates
                (heading ?d - direction)
                (clockwise ?d1 ?d2 - direction)
                (at ?x - coord ?y - coord)
                (nextx ?a - coord ?b - coord ?h - direction)
                (nexty ?a - coord ?b - coord ?h - direction)
                (light_color ?c - color)
                (light_delay ?x ?y - coord ?d - delay)
                (light_preference ?x ?y - coord ?p - preference)
                (road-length ?start-x ?start-y ?end-x ?end-y - coord ?l - length)
                (alive)
        )

        (:constants
                ;; deprected
                left right straight - rotation

                north south east west - direction
                green red unknown - color
                north_south none east_west - preference
                quick normal slow - delay
                short medium long - length
        )

        (:functions (total-cost))

        (:action look_at_light-north-north_south
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading north)
                                (light_preference ?x ?y north_south)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             9/10
                                   (and (not (light_color unknown))(light_color green))
                             1/10
                                    (and (not (light_color unknown))(light_color red))
                            ))
        )

        (:action look_at_light-south-north_south
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading south)
                                (light_preference ?x ?y north_south)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             9/10
                                   (and (not (light_color unknown))(light_color gree))
                             1/10
                                   (and (not (light_color unknown))(light_color red))
                            )
                        )
        )

        (:action look_at_light-east-north_south
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading east)
                                (light_preference ?x ?y north_south)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             1/10
                                   (and (not (light_color unknown))(light_color green))
                             9/10
                                   (and (not (light_color unknown))(light_color red))))

        )

        (:action look_at_light-west-north_south
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading west)
                                (light_preference ?x ?y north_south)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             1/10
                                    (and (not (light_color unknown))(light_color green))
                             9/10
                                   (and (not (light_color unknown))(light_color red))))
        )

        (:action look_at_light-north-east_west
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading north)
                                (light_preference ?x ?y east_west)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             1/10
                                   (and (not (light_color unknown))(light_color green))
                             9/10
                                   (and (not (light_color unknown))(light_color red))))
        )

        (:action look_at_light-south-east_west
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading south)
                                (light_preference ?x ?y east_west)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             1/10
                                  (and (not (light_color unknown))(light_color green))
                             9/10
                                  (and (not (light_color unknown))(light_color red))))
        )

        (:action look_at_light-east-east_west
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading east)
                                (light_preference ?x ?y east_west)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             9/10
                                   (and (not (light_color unknown))(light_color green))
                             1/10
                                   (and (not (light_color unknown))(light_color red))))
        )

        (:action look_at_light-west-east_west
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (heading west)
                                (light_preference ?x ?y east_west)
                        )
                :effect (and (increase (total-cost) 1)
                           (probabilistic
                             9/10
                                   (and (not (light_color unknown))(light_color green))
                             1/10
                                   (and (not (light_color unknown))(light_color red))))
        )

        (:action look_at_light
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color unknown)
                                (at ?x ?y)
                                (light_preference ?x ?y none)
                        )
                :effect (and (increase (total-cost) 1)
                            (probabilistic
                             1/2
                                   (and (not (light_color unknown))(light_color green))
                             1/2
                                   (and (not (light_color unknown))(light_color red))))
        )

        (:action wait_on_light-quick
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color red)
                                (at ?x ?y)
                                (light_delay ?x ?y quick)
                        )
                :effect (and (increase (total-cost) 1)
                            (probabilistic
                                1/100 (not (alive))
                            )

                            (probabilistic
                                1/2 (and (not (light_color red))(light_color green)))
                        )
        )

        (:action wait_on_light-normal
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color red)
                                (at ?x ?y)
                                (light_delay ?x ?y normal)
                        )
                :effect (and (increase (total-cost) 1)
                            (probabilistic
                                1/100 (not (alive))
                            )

                            (probabilistic
                              1/5
                                    (and (not (light_color red))(light_color green)))
                        )
        )

        (:action wait_on_light-long
                :parameters (?x - coord ?y - coord)
                :precondition (and
                                (light_color red)
                                (at ?x ?y)
                                (light_delay ?x ?y quick)
                        )
                :effect (and (increase (total-cost) 1)
                            (probabilistic
                                1/100 (not (alive))
                            )

                            (probabilistic
                              1/10
                                    (and (not (light_color red))(light_color green)))
                        )
        )

        (:action proceed-short-left
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (clockwise ?new-heading ?old-heading)
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y short))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/50 (not (alive)))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )

        (:action proceed-medium-left
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (clockwise ?new-heading ?old-heading)
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y medium))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/20 (not (alive)))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )

        (:action proceed-long-left
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (clockwise ?new-heading ?old-heading)
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y long))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/10 (not (alive)))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )

        (:action proceed-short-right
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (clockwise ?old-heading ?new-heading)
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y short))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/50 (not (alive)))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )

        (:action proceed-medium-right
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (clockwise ?old-heading ?new-heading)
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y medium))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/20 (not (alive)))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )

        (:action proceed-long-right
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?old-heading ?new-heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?old-heading)
                                (clockwise ?old-heading ?new-heading)
                                (nextx ?x ?new-x ?new-heading)
                                (nexty ?y ?new-y ?new-heading)
                                (road-length ?x ?y ?new-x ?new-y long))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/10 (not (alive)))
                                (not (at ?x ?y))
                                (not (heading ?old-heading))
                                (heading ?new-heading)
                                (at ?new-x ?new-y))
        )

        (:action proceed-short-straight
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?heading)
                                (nextx ?x ?new-x ?heading)
                                (nexty ?y ?new-y ?heading)
                                (road-length ?x ?y ?new-x ?new-y short))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (not (at ?x ?y))
                                (at ?new-x ?new-y)
                                (probabilistic 1/50 (not (alive)))
                        )
        )

        (:action proceed-medium-straight
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?heading)
                                (nextx ?x ?new-x ?heading)
                                (nexty ?y ?new-y ?heading)
                                (road-length ?x ?y ?new-x ?new-y medium))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/20 (not (alive)))
                                (not (at ?x ?y))
                                (at ?new-x ?new-y))
        )

        (:action proceed-short-straight
                :parameters (?x ?y ?new-x ?new-y - coord
                             ?heading - direction)
                :precondition (and
                                (light_color green)
                                (at ?x ?y)
                                (heading ?heading)
                                (nextx ?x ?new-x ?heading)
                                (nexty ?y ?new-y ?heading)
                                (road-length ?x ?y ?new-x ?new-y long))
                :effect (and (increase (total-cost) 1)
                                (not (light_color green))
                                (light_color unknown)
                                (probabilistic 1/10 (not (alive)))
                                (not (at ?x ?y))
                                (at ?new-x ?new-y))
        )
)
