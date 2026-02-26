 for 2. maybe we need to write a tool that makes it easy (convenient manual workflow) to verify the    
  beat detection and manually choose the downbeat. Like space for playback one bar, l for next bar, h     
  for previous bar, d for delete (reomve from data labels), and enter for confirm beat (tempo) and        
  downbeat. minimal UI. That then creates labeld data that can later be chunked.                          
                                                                                                          
  When user can label 2 songs per minute like this, an hour of work wold be 120 songs => 360 chunks (or   
  with even more augmentation: pitch, noise, tempo changes > 1000 chunkgs worth of refining the midi      
  data trained model - sounds good!)                                                                    
                                                                                                          
  ---                                                                                                     
                                                                                                          
  for 1.: I like your 4 types of tempo augmentation and we should allow them to happen at any point (mid  
  chunk. Maybe we pull apart the rendering and the chunk generation again so that we can treat midi       
  generated labeld data and manually labelled realworld audio the same from (inlc.) chunking.             
                                                                                                          
                                                                                                          
  Create to files in meta: realworld_audio_training_data.md and                                           
  augmentation_refactor_and_tempo_augmentation.md                                                         
                                                                                                          
                                                                                                          
  1. no. manually confirmed mostly automated downbeat estimation as described above                       
  2. i think 0.5 to 2.0 should be maximum for abrupt tempo changes. (half or double)                      
  3. both matter.                                                                                         
  4. like i described above, let's make the process 95% automatic + 5% human (convenient) for             
  confirmation/quality control. Output should be stored as original adio file + meta info (phase data -   
  maybe in form of phase start points instead of samples so that we have continous resolution)            
  5. they shall coexist and compound. let's just be careful that the phase labels are always correct in   
  the end.                                   
