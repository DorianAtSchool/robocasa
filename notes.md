The trajectories in run sameple trajectories are missing an important part which is the id of the agent (i.e. 0, 1) indicating which      
  robot is commuinicating or executing the action.                                                                                    
                                                     
  Something here is a little confusing because we generate a combined trajectory but we have to train a vlm on individual trajectories as
  otherwise it wouldnt make sense to communicate, right? or am i missing something? I guess we could split up the trajectory, but im not   
  sure if that makes sense.                                                        
                                                                                                                    
  The robots need to be teleported with the items and towards items they are itneracting with for all views to match
                                                      
  Can we create a file to see the simulation in action
                                                                                                                                       
  Also we need a way to distinguish between the robots, maybe by color - it should be possible to create new usd files with diff colors
  for robots or to color it in sim directly        