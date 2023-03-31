Device NVIDIA A100 80GB PCIe (GA100)
--------------------------------------------------------------------------- --------------- --------------- ----------------------------------------------------------------------
Metric Name                                                                 Metric Type     Metric Unit     Metric Description                                                    
--------------------------------------------------------------------------- --------------- --------------- ----------------------------------------------------------------------
dram__bytes                                                                 Counter         byte            # of bytes accessed in DRAM                                           
dram__bytes_read                                                            Counter         byte            # of bytes read from DRAM                                             
dram__bytes_write                                                           Counter         byte            # of bytes written to DRAM                                            
dram__cycles_active                                                         Counter         cycle           # of cycles where DRAM was active                                     
dram__cycles_active_read                                                    Counter         cycle           # of cycles where DRAM was active for reads                           
dram__cycles_active_write                                                   Counter         cycle           # of cycles where DRAM was active for writes                          
dram__cycles_elapsed                                                        Counter         cycle           # of elapsed DRAM memory clock cycles                                 
dram__cycles_in_frame                                                       Counter         cycle           # of cycles in user-defined frame                                     
dram__cycles_in_region                                                      Counter         cycle           # of cycles in user-defined region                                    
dram__sectors                                                               Counter         sector          # of sectors accessed in DRAM                                         
dram__sectors_read                                                          Counter         sector          # of sectors read from DRAM                                           
dram__sectors_write                                                         Counter         sector          # of sectors written to DRAM                                          
dram__throughput                                                            Throughput      %               DRAM throughput                                                       
fbpa__cycles_active                                                         Counter         cycle           # of cycles where FBPA                                                
fbpa__cycles_elapsed                                                        Counter         cycle           # of cycles elapsed on FBPA                                           
fbpa__cycles_in_frame                                                       Counter         cycle           # of cycles in user-defined frame                                     
fbpa__cycles_in_region                                                      Counter         cycle           # of cycles in user-defined region                                    
fbpa__dram_cycles_elapsed                                                   Counter         cycle           # of cycles elapsed on DRAM                                           
fbpa__dram_read_bytes                                                       Counter         byte            # of DRAM read bytes                                                  
fbpa__dram_read_sectors                                                     Counter         sector          # of DRAM read sectors                                                
fbpa__dram_read_throughput                                                  Throughput      %               FBPA read throughput                                                  
fbpa__dram_sectors                                                          Counter         sector          # of DRAM sectors accessed                                            
fbpa__dram_write_bytes                                                      Counter         byte            # of DRAM write bytes                                                 
fbpa__dram_write_sectors                                                    Counter         sector          # of DRAM write sectors                                               
fbpa__dram_write_throughput                                                 Throughput      %               FBPA write throughput                                                 
fbpa__throughput                                                            Throughput      %               FBPA throughput                                                       
fe__cycles_active                                                           Counter         cycle           alias of fe__cycles_elapsed                                           
fe__cycles_elapsed                                                          Counter         cycle           # of cycles elapsed on FE                                             
fe__cycles_in_frame                                                         Counter         cycle           # of cycles in user-defined frame                                     
fe__cycles_in_region                                                        Counter         cycle           # of cycles in user-defined region                                    
fe__output_ops_type_bundle_cmd_go_idle                                      Counter                         # of GO_IDLE bundles sent to GR from FE                               
gpc__cycles_active                                                          Counter         cycle           # of cycles where GPC was active                                      
gpc__cycles_elapsed                                                         Counter         cycle           # of cycles elapsed on GPC                                            
gpc__cycles_in_frame                                                        Counter         cycle           # of cycles in user-defined frame                                     
gpc__cycles_in_region                                                       Counter         cycle           # of cycles in user-defined region                                    
gpu__compute_memory_access_throughput                                       Throughput      %               Compute Memory Pipeline : throughput of internal activity within      
                                                                                                            caches and DRAM                                                       
gpu__compute_memory_access_throughput_internal_activity                     Throughput      %               Compute Memory Pipeline : throughput of internal activity within      
                                                                                                            caches and DRAM, internal activity                                    
gpu__compute_memory_request_throughput                                      Throughput      %               Compute Memory Pipeline : throughput of interconnects between         
                                                                                                            SM<->Caches<->DRAM                                                    
gpu__compute_memory_request_throughput_internal_activity                    Throughput      %               Compute Memory Pipeline : throughput of interconnects between         
                                                                                                            SM<->Caches<->DRAM, internal activity                                 
gpu__compute_memory_throughput                                              Throughput      %               Compute Memory Pipeline Throughput                                    
gpu__cycles_active                                                          Counter         cycle           # of cycles where GPU was active                                      
gpu__cycles_elapsed                                                         Counter         cycle           # of cycles elapsed on GPU                                            
gpu__cycles_in_frame                                                        Counter         cycle           # of cycles in user-defined frame                                     
gpu__cycles_in_region                                                       Counter         cycle           # of cycles in user-defined region                                    
gpu__dram_throughput                                                        Throughput      %               GPU DRAM throughput                                                   
gpu__time_active                                                            Counter         nsecond         total duration in nanoseconds                                         
gpu__time_duration                                                          Counter         nsecond         equals to gpu__time_duration_measured_user if collectable, otherwise  
                                                                                                            equals to gpu__time_duration_measured_wallclock                       
gpu__time_duration_measured_user                                            Counter         nsecond         context-aware time duration in nanoseconds                            
gpu__time_duration_measured_wallclock                                       Counter         nsecond         wall-clock duration in nanoseconds                                    
gpu__time_end                                                               Counter         nsecond         end timestamp, relative to start of frame                             
gpu__time_start                                                             Counter         nsecond         start timestamp, relative to start of frame                           
gr__cycles_active                                                           Counter         cycle           # of cycles where GR was active                                       
gr__cycles_elapsed                                                          Counter         cycle           # of cycles elapsed on GR                                             
gr__cycles_idle                                                             Counter         cycle           # of cycles where GR was idle                                         
gr__cycles_in_frame                                                         Counter         cycle           # of cycles in user-defined frame                                     
gr__cycles_in_region                                                        Counter         cycle           # of cycles in user-defined region                                    
idc__cycles_active                                                          Counter         cycle           # of cycles where IDC was active                                      
idc__cycles_elapsed                                                         Counter         cycle           # of cycles elapsed on IDC                                            
idc__cycles_in_frame                                                        Counter         cycle           # of cycles in user-defined frame                                     
idc__cycles_in_region                                                       Counter         cycle           # of cycles in user-defined region                                    
idc__request_cycles_active                                                  Counter         cycle           # of cycles where IDC processed requests from SM                      
idc__request_hit_rate                                                       Ratio                           proportion of IDC requests that hit                                   
idc__requests                                                               Counter                         # of IDC requests                                                     
idc__requests_lookup_hit                                                    Counter                         # of IDC requests that hit                                            
idc__requests_lookup_miss                                                   Counter                         # of IDC requests that missed                                         
l1tex__average_t_sectors_per_request                                        Ratio           sector/request  average # of sectors requested per request sent to T stage            
l1tex__average_t_sectors_per_request_pipe_lsu                               Ratio           sector/request  average # of sectors requested per request sent to T stage for LSU    
                                                                                                            pipe local/global                                                     
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_atom            Ratio           sector/request  average # of sectors requested per request sent to T stage for global 
                                                                                                            atomics                                                               
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld              Ratio           sector/request  average # of sectors requested per request sent to T stage for global 
                                                                                                            loads                                                                 
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_red             Ratio           sector/request  average # of sectors requested per request sent to T stage for global 
                                                                                                            reductions                                                            
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st              Ratio           sector/request  average # of sectors requested per request sent to T stage for global 
                                                                                                            stores                                                                
l1tex__average_t_sectors_per_request_pipe_lsu_mem_local_op_ld               Ratio           sector/request  average # of sectors requested per request sent to T stage for local  
                                                                                                            loads                                                                 
l1tex__average_t_sectors_per_request_pipe_lsu_mem_local_op_st               Ratio           sector/request  average # of sectors requested per request sent to T stage for local  
                                                                                                            stores                                                                
l1tex__average_t_sectors_per_request_pipe_tex                               Ratio           sector/request  average # of sectors requested per request sent to T stage for TEX    
                                                                                                            pipe                                                                  
l1tex__average_t_sectors_per_request_pipe_tex_mem_surface                   Ratio           sector/request  average # of sectors requested per request sent to T stage for surface
l1tex__average_t_sectors_per_request_pipe_tex_mem_surface_op_atom           Ratio           sector/request  average # of sectors requested per request sent to T stage for        
                                                                                                            surface atomics                                                       
l1tex__average_t_sectors_per_request_pipe_tex_mem_surface_op_ld             Ratio           sector/request  average # of sectors requested per request sent to T stage for        
                                                                                                            surface loads                                                         
l1tex__average_t_sectors_per_request_pipe_tex_mem_surface_op_red            Ratio           sector/request  average # of sectors requested per request sent to T stage for        
                                                                                                            surface reductions                                                    
l1tex__average_t_sectors_per_request_pipe_tex_mem_surface_op_st             Ratio           sector/request  average # of sectors requested per request sent to T stage for        
                                                                                                            surface stores                                                        
l1tex__average_t_sectors_per_request_pipe_tex_mem_texture                   Ratio           sector/request  average # of sectors requested per request sent to T stage for texture
l1tex__average_t_sectors_per_request_pipe_tex_mem_texture_op_ld             Ratio           sector/request  average # of sectors requested per request sent to T stage for TLD    
                                                                                                            instructions                                                          
l1tex__average_t_sectors_per_request_pipe_tex_mem_texture_op_tex            Ratio           sector/request  average # of sectors requested per request sent to T stage for TEX    
                                                                                                            instructions                                                          
l1tex__cycles_active                                                        Counter         cycle           # of cycles where L1TEX was active                                    
l1tex__cycles_elapsed                                                       Counter         cycle           # of cycles elapsed on L1TEX                                          
l1tex__cycles_in_frame                                                      Counter         cycle           # of cycles in user-defined frame                                     
l1tex__cycles_in_region                                                     Counter         cycle           # of cycles in user-defined region                                    
l1tex__data_bank_conflicts_pipe_lsu                                         Counter                         # of data bank conflicts generated by LSU pipe                        
l1tex__data_bank_conflicts_pipe_lsu_cmd_read                                Counter                         # of data bank conflicts generated by LSU reads                       
l1tex__data_bank_conflicts_pipe_lsu_cmd_write                               Counter                         # of data bank conflicts generated by LSU writes                      
l1tex__data_bank_conflicts_pipe_lsu_mem_global                              Counter                         # of data bank conflicts generated by global ops                      
l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_atom                      Counter                         # of data bank conflicts generated by global atomics                  
l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_ld                        Counter                         # of data bank conflicts generated by global loads                    
l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_red                       Counter                         # of data bank conflicts generated by global reductions               
l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st                        Counter                         # of data bank conflicts generated by global stores                   
l1tex__data_bank_conflicts_pipe_lsu_mem_shared                              Counter                         # of shared memory data bank conflicts generated by LDS, LD, 3D       
                                                                                                            attribute loads, LDSM, STS, ST, ATOMS, ATOM, 3D attribute stores,     
                                                                                                            LDGSTS and Misc.                                                      
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom                      Counter                         # of shared memory data bank conflicts generated by ATOMS, ATOM       
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld                        Counter                         # of shared memory data bank conflicts generated by LDS, LD, 3D       
                                                                                                            attribute loads, LDSM                                                 
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts                    Counter                         # of data bank conflicts generated by shared LDGSTS ops               
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st                        Counter                         # of shared memory data bank conflicts generated by STS, ST, 3D       
                                                                                                            attribute stores                                                      
l1tex__data_bank_reads                                                      Counter                         # of data bank reads                                                  
l1tex__data_bank_writes                                                     Counter                         # of data bank writes                                                 
l1tex__data_pipe_lsu_wavefronts                                             Counter                         # of local/global/shared + surface write wavefronts processed by      
                                                                                                            Data-Stage                                                            
l1tex__data_pipe_lsu_wavefronts_cmd_read                                    Counter                         # of local/global/shared read wavefronts processed by Data-Stage      
l1tex__data_pipe_lsu_wavefronts_cmd_write                                   Counter                         # of local/global/shared/surface write wavefronts processed by        
                                                                                                            Data-Stage                                                            
l1tex__data_pipe_lsu_wavefronts_mem_lg                                      Counter                         # of local/global wavefronts processed by Data-Stage                  
l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_read                             Counter                         # of local/global read wavefronts processed by Data-Stage             
l1tex__data_pipe_lsu_wavefronts_mem_lg_cmd_write                            Counter                         # of local/global write wavefronts processed by Data-Stage            
l1tex__data_pipe_lsu_wavefronts_mem_shared                                  Counter                         # of shared memory wavefronts processed by Data-Stage for LDS, LD, 3D 
                                                                                                            attribute loads, LDSM, STS, ST, ATOMS, ATOM, 3D attribute stores,     
                                                                                                            LDGSTS and Misc.                                                      
l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_read                         Counter                         # of shared read wavefronts processed by Data-Stage                   
l1tex__data_pipe_lsu_wavefronts_mem_shared_cmd_write                        Counter                         # of shared write wavefronts processed by Data-Stage                  
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_atom                          Counter                         # of shared memory wavefronts processed by Data-Stage for ATOMS, ATOM 
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ipa                           Counter                         # of IPA wavefronts processed by Data-Stage                           
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld                            Counter                         # of shared memory wavefronts processed by Data-Stage for LDS, LD, 3D 
                                                                                                            attribute loads, LDSM                                                 
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st                            Counter                         # of shared memory wavefronts processed by Data-Stage for STS, ST, 3D 
                                                                                                            attribute stores                                                      
l1tex__data_pipe_lsu_wavefronts_mem_surface                                 Counter                         # of surface write wavefronts processed by Data-Stage                 
l1tex__data_pipe_tex_wavefronts                                             Counter                         # of texture + surface read wavefronts processed by Data-Stage        
l1tex__data_pipe_tex_wavefronts_mem_surface                                 Counter                         # of surface read wavefronts processed by Data-Stage                  
l1tex__data_pipe_tex_wavefronts_mem_texture                                 Counter                         # of texture wavefronts processed by Data-Stage                       
l1tex__data_pipe_tex_wavefronts_realtime                                    Counter                         # of texture + surface wavefronts processed by Data-Stage             
l1tex__f_cycles_active                                                      Counter                         # of cycles where L1TEX F-Stage was active                            
l1tex__f_tex2sm_cycles_active                                               Counter         cycle           # of cycles where interface carrying return data from L1TEX F-Stage   
                                                                                                            back to SM was active                                                 
l1tex__f_tex2sm_cycles_stalled                                              Counter         cycle           # of cycles where interface carrying return data from L1TEX F-Stage   
                                                                                                            back to SM was stalled                                                
l1tex__f_wavefronts                                                         Counter                         # of wavefronts processed by L1TEX F-Stage                            
l1tex__lsu_writeback_active                                                 Counter         cycle           # of cycles where local/global/shared writeback interface was active  
l1tex__lsu_writeback_active_mem_lg                                          Counter         cycle           # of cycles where local/global writeback interface was active         
l1tex__lsu_writeback_active_mem_shared                                      Counter         cycle           # of cycles where local/global/shared writeback interface was active  
                                                                                                            for shared memory instructions                                        
l1tex__lsuin_requests                                                       Counter         request         # of local/global/shared/attribute instructions sent to LSU           
l1tex__m_l1tex2xbar_req_cycles_active                                       Counter         cycle           # of cycles where interface from L1TEX M-Stage to XBAR was active     
l1tex__m_l1tex2xbar_req_cycles_stalled                                      Counter         cycle           # of cycles where interface from L1TEX M-Stage to XBAR was stalled    
l1tex__m_l1tex2xbar_throughput                                              Throughput      %               L1TEX M-Stage to XBAR throughput                                      
l1tex__m_l1tex2xbar_throughput_pipe_lsu                                     Throughput      %               L1TEX M-Stage to XBAR throughput for LSU pipe                         
l1tex__m_l1tex2xbar_throughput_pipe_tex                                     Throughput      %               L1TEX M-Stage to XBAR throughput for TEX pipe                         
l1tex__m_l1tex2xbar_write_bytes                                             Counter         byte            # of bytes written to L2                                              
l1tex__m_l1tex2xbar_write_bytes_mem_global_op_atom                          Counter         byte            # of bytes written to L2 for global atomics                           
l1tex__m_l1tex2xbar_write_bytes_mem_global_op_red                           Counter         byte            # of bytes written to L2 for global reductions                        
l1tex__m_l1tex2xbar_write_bytes_mem_lg_op_st                                Counter         byte            # of bytes written to L2 for local/global stores                      
l1tex__m_l1tex2xbar_write_bytes_mem_surface_op_atom                         Counter         byte            # of bytes written to L2 for surface atomics                          
l1tex__m_l1tex2xbar_write_bytes_mem_surface_op_red                          Counter         byte            # of bytes written to L2 for surface reductions                       
l1tex__m_l1tex2xbar_write_bytes_mem_surface_op_st                           Counter         byte            # of bytes written to L2 for surface stores                           
l1tex__m_l1tex2xbar_write_bytes_pipe_lsu                                    Counter         byte            # of bytes written to L2 for LSU pipe                                 
l1tex__m_l1tex2xbar_write_bytes_pipe_tex                                    Counter         byte            # of bytes written to L2 for TEX pipe                                 
l1tex__m_l1tex2xbar_write_sectors                                           Counter         sector          # of sectors written to L2                                            
l1tex__m_l1tex2xbar_write_sectors_mem_global_op_atom                        Counter         sector          # of sectors written to L2 for global atomics                         
l1tex__m_l1tex2xbar_write_sectors_mem_global_op_red                         Counter         sector          # of sectors written to L2 for global reductions                      
l1tex__m_l1tex2xbar_write_sectors_mem_lg_op_st                              Counter         sector          # of sectors written to L2 for local/global stores                    
l1tex__m_l1tex2xbar_write_sectors_mem_surface_op_atom                       Counter         sector          # of sectors written to L2 for surface atomics                        
l1tex__m_l1tex2xbar_write_sectors_mem_surface_op_red                        Counter         sector          # of sectors written to L2 for surface reductions                     
l1tex__m_l1tex2xbar_write_sectors_mem_surface_op_st                         Counter         sector          # of sectors written to L2 for surface stores                         
l1tex__m_l1tex2xbar_write_sectors_pipe_lsu                                  Counter         sector          # of sectors written to L2 for LSU pipe                               
l1tex__m_l1tex2xbar_write_sectors_pipe_tex                                  Counter         sector          # of sectors written to L2 for TEX pipe                               
l1tex__m_xbar2l1tex_read_bytes                                              Counter         byte            # of bytes read from L2 into L1TEX M-Stage                            
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_atom                           Counter         byte            # of bytes read from L2 into L1TEX M-Stage for global atomics         
l1tex__m_xbar2l1tex_read_bytes_mem_lg_op_ld                                 Counter         byte            # of bytes read from L2 into L1TEX M-Stage for local/global loads     
l1tex__m_xbar2l1tex_read_bytes_mem_surface_op_atom                          Counter         byte            # of bytes read from L2 into L1TEX M-Stage for surface atomics        
l1tex__m_xbar2l1tex_read_bytes_mem_surface_op_ld                            Counter         byte            # of bytes read from L2 into L1TEX M-Stage for surface loads          
l1tex__m_xbar2l1tex_read_bytes_mem_texture                                  Counter         byte            # of bytes read from L2 into L1TEX M-Stage for texture                
l1tex__m_xbar2l1tex_read_bytes_pipe_lsu                                     Counter         byte            # of bytes read from L2 into L1TEX M-Stage for LSU pipe               
l1tex__m_xbar2l1tex_read_bytes_pipe_tex                                     Counter         byte            # of bytes read from L2 into L1TEX M-Stage for TEX pipe               
l1tex__m_xbar2l1tex_read_sectors                                            Counter         sector          # of sectors read from L2 into L1TEX M-Stage                          
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_atom                         Counter         sector          # of sectors read from L2 into L1TEX M-Stage for global atomics       
l1tex__m_xbar2l1tex_read_sectors_mem_lg_op_ld                               Counter         sector          # of sectors read from L2 into L1TEX M-Stage for local/global loads   
l1tex__m_xbar2l1tex_read_sectors_mem_surface_op_atom                        Counter         sector          # of sectors read from L2 into L1TEX M-Stage for surface atomics      
l1tex__m_xbar2l1tex_read_sectors_mem_surface_op_ld                          Counter         sector          # of sectors read from L2 into L1TEX M-Stage for surface loads        
l1tex__m_xbar2l1tex_read_sectors_mem_texture                                Counter         sector          # of sectors read from L2 into L1TEX M-Stage for texture              
l1tex__m_xbar2l1tex_read_sectors_pipe_lsu                                   Counter         sector          # of sectors read from L2 into L1TEX M-Stage for LSU pipe             
l1tex__m_xbar2l1tex_read_sectors_pipe_tex                                   Counter         sector          # of sectors read from L2 into L1TEX M-Stage for TEX pipe             
l1tex__m_xbar2l1tex_throughput                                              Throughput      %               XBAR to L1TEX M-Stage throughput                                      
l1tex__m_xbar2l1tex_throughput_pipe_lsu                                     Throughput      %               XBAR to L1TEX M-Stage throughput for LSU pipe                         
l1tex__m_xbar2l1tex_throughput_pipe_tex                                     Throughput      %               XBAR to L1TEX M-Stage throughput for TEX pipe                         
l1tex__t_bytes                                                              Counter         byte            # of bytes requested                                                  
l1tex__t_bytes_lookup_hit                                                   Counter         byte            # of bytes requested with tag-hit and data-hit                        
l1tex__t_bytes_lookup_miss                                                  Counter         byte            # of bytes requested that missed                                      
l1tex__t_bytes_pipe_lsu                                                     Counter         byte            # of bytes requested for LSU pipe local/global                        
l1tex__t_bytes_pipe_lsu_lookup_hit                                          Counter         byte            # of bytes requested with tag-hit and data-hit for LSU pipe           
                                                                                                            local/global                                                          
l1tex__t_bytes_pipe_lsu_lookup_miss                                         Counter         byte            # of bytes requested that missed for LSU pipe local/global            
l1tex__t_bytes_pipe_lsu_mem_global_op_atom                                  Counter         byte            # of bytes requested for global atomics                               
l1tex__t_bytes_pipe_lsu_mem_global_op_atom_lookup_hit                       Counter         byte            # of bytes requested with tag-hit and data-hit for global atomics     
l1tex__t_bytes_pipe_lsu_mem_global_op_atom_lookup_miss                      Counter         byte            # of bytes requested that missed for global atomics                   
l1tex__t_bytes_pipe_lsu_mem_global_op_ld                                    Counter         byte            # of bytes requested for global loads                                 
l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_hit                         Counter         byte            # of bytes requested with tag-hit and data-hit for global loads       
l1tex__t_bytes_pipe_lsu_mem_global_op_ld_lookup_miss                        Counter         byte            # of bytes requested that missed for global loads                     
l1tex__t_bytes_pipe_lsu_mem_global_op_red                                   Counter         byte            # of bytes requested for global reductions                            
l1tex__t_bytes_pipe_lsu_mem_global_op_red_lookup_hit                        Counter         byte            # of bytes requested with tag-hit and data-hit for global reductions  
l1tex__t_bytes_pipe_lsu_mem_global_op_red_lookup_miss                       Counter         byte            # of bytes requested that missed for global reductions                
l1tex__t_bytes_pipe_lsu_mem_global_op_st                                    Counter         byte            # of bytes requested for global stores                                
l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_hit                         Counter         byte            # of bytes requested with tag-hit and data-hit for global stores      
l1tex__t_bytes_pipe_lsu_mem_global_op_st_lookup_miss                        Counter         byte            # of bytes requested that missed for global stores                    
l1tex__t_bytes_pipe_lsu_mem_local_op_ld                                     Counter         byte            # of bytes requested for local loads                                  
l1tex__t_bytes_pipe_lsu_mem_local_op_ld_lookup_hit                          Counter         byte            # of bytes requested with tag-hit and data-hit for local loads        
l1tex__t_bytes_pipe_lsu_mem_local_op_ld_lookup_miss                         Counter         byte            # of bytes requested that missed for local loads                      
l1tex__t_bytes_pipe_lsu_mem_local_op_st                                     Counter         byte            # of bytes requested for local stores                                 
l1tex__t_bytes_pipe_lsu_mem_local_op_st_lookup_hit                          Counter         byte            # of bytes requested with tag-hit and data-hit for local stores       
l1tex__t_bytes_pipe_lsu_mem_local_op_st_lookup_miss                         Counter         byte            # of bytes requested that missed for local stores                     
l1tex__t_bytes_pipe_tex                                                     Counter         byte            # of bytes requested for TEX pipe                                     
l1tex__t_bytes_pipe_tex_lookup_hit                                          Counter         byte            # of bytes requested with tag-hit and data-hit for TEX pipe           
l1tex__t_bytes_pipe_tex_lookup_miss                                         Counter         byte            # of bytes requested that missed for TEX pipe                         
l1tex__t_bytes_pipe_tex_mem_surface                                         Counter         byte            # of bytes requested for surface                                      
l1tex__t_bytes_pipe_tex_mem_surface_lookup_hit                              Counter         byte            # of bytes requested with tag-hit and data-hit for surface            
l1tex__t_bytes_pipe_tex_mem_surface_lookup_miss                             Counter         byte            # of bytes requested that missed for surface                          
l1tex__t_bytes_pipe_tex_mem_surface_op_atom                                 Counter         byte            # of bytes requested for surface atomics                              
l1tex__t_bytes_pipe_tex_mem_surface_op_atom_lookup_hit                      Counter         byte            # of bytes requested with tag-hit and data-hit for surface atomics    
l1tex__t_bytes_pipe_tex_mem_surface_op_atom_lookup_miss                     Counter         byte            # of bytes requested that missed for surface atomics                  
l1tex__t_bytes_pipe_tex_mem_surface_op_ld                                   Counter         byte            # of bytes requested for surface loads                                
l1tex__t_bytes_pipe_tex_mem_surface_op_ld_lookup_hit                        Counter         byte            # of bytes requested with tag-hit and data-hit for surface loads      
l1tex__t_bytes_pipe_tex_mem_surface_op_ld_lookup_miss                       Counter         byte            # of bytes requested that missed for surface loads                    
l1tex__t_bytes_pipe_tex_mem_surface_op_red                                  Counter         byte            # of bytes requested for surface reductions                           
l1tex__t_bytes_pipe_tex_mem_surface_op_red_lookup_hit                       Counter         byte            # of bytes requested with tag-hit and data-hit for surface reductions 
l1tex__t_bytes_pipe_tex_mem_surface_op_red_lookup_miss                      Counter         byte            # of bytes requested that missed for surface reductions               
l1tex__t_bytes_pipe_tex_mem_surface_op_st                                   Counter         byte            # of bytes requested for surface stores                               
l1tex__t_bytes_pipe_tex_mem_surface_op_st_lookup_hit                        Counter         byte            # of bytes requested with tag-hit and data-hit for surface stores     
l1tex__t_bytes_pipe_tex_mem_surface_op_st_lookup_miss                       Counter         byte            # of bytes requested that missed for surface stores                   
l1tex__t_bytes_pipe_tex_mem_texture                                         Counter         byte            # of bytes requested for texture                                      
l1tex__t_bytes_pipe_tex_mem_texture_lookup_hit                              Counter         byte            # of bytes requested with tag-hit and data-hit for texture            
l1tex__t_bytes_pipe_tex_mem_texture_lookup_miss                             Counter         byte            # of bytes requested that missed for texture                          
l1tex__t_bytes_pipe_tex_mem_texture_op_ld                                   Counter         byte            # of bytes requested for TLD instructions                             
l1tex__t_bytes_pipe_tex_mem_texture_op_ld_lookup_hit                        Counter         byte            # of bytes requested with tag-hit and data-hit for TLD instructions   
l1tex__t_bytes_pipe_tex_mem_texture_op_ld_lookup_miss                       Counter         byte            # of bytes requested that missed for TLD instructions                 
l1tex__t_bytes_pipe_tex_mem_texture_op_tex                                  Counter         byte            # of bytes requested for TEX instructions                             
l1tex__t_bytes_pipe_tex_mem_texture_op_tex_lookup_hit                       Counter         byte            # of bytes requested with tag-hit and data-hit for TEX instructions   
l1tex__t_bytes_pipe_tex_mem_texture_op_tex_lookup_miss                      Counter         byte            # of bytes requested that missed for TEX instructions                 
l1tex__t_output_wavefronts                                                  Counter                         # of wavefronts sent to Data-Stage from T-Stage                       
l1tex__t_output_wavefronts_pipe_lsu                                         Counter                         # of wavefronts sent to Data-Stage from T-Stage for LSU pipe          
l1tex__t_output_wavefronts_pipe_lsu_mem_global                              Counter                         # of wavefronts sent to Data-Stage from T-Stage for global memory     
                                                                                                            instructions                                                          
l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_atom                      Counter                         # of wavefronts sent to Data-Stage from T-Stage for global atomics    
l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld                        Counter                         # of wavefronts sent to Data-Stage from T-Stage for global loads      
l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_red                       Counter                         # of wavefronts sent to Data-Stage from T-Stage for global reductions 
l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_st                        Counter                         # of wavefronts sent to Data-Stage from T-Stage for global stores     
l1tex__t_output_wavefronts_pipe_lsu_mem_local                               Counter                         # of wavefronts sent to Data-Stage from T-Stage for local memory      
                                                                                                            instructions                                                          
l1tex__t_output_wavefronts_pipe_lsu_mem_local_op_ld                         Counter                         # of wavefronts sent to Data-Stage from T-Stage for local loads       
l1tex__t_output_wavefronts_pipe_lsu_mem_local_op_st                         Counter                         # of wavefronts sent to Data-Stage from T-Stage for local stores      
l1tex__t_output_wavefronts_pipe_tex                                         Counter                         # of wavefronts sent to Data-Stage from T-Stage for TEX pipe          
l1tex__t_output_wavefronts_pipe_tex_mem_surface                             Counter                         # of wavefronts sent to Data-Stage from T-Stage for surface           
                                                                                                            instructions                                                          
l1tex__t_output_wavefronts_pipe_tex_mem_surface_op_atom                     Counter                         # of wavefronts sent to Data-Stage from T-Stage for surface atomics   
l1tex__t_output_wavefronts_pipe_tex_mem_surface_op_ld                       Counter                         # of wavefronts sent to Data-Stage from T-Stage for surface loads     
l1tex__t_output_wavefronts_pipe_tex_mem_surface_op_red                      Counter                         # of wavefronts sent to Data-Stage from T-Stage for surface reductions
l1tex__t_output_wavefronts_pipe_tex_mem_surface_op_st                       Counter                         # of wavefronts sent to Data-Stage from T-Stage for surface stores    
l1tex__t_output_wavefronts_pipe_tex_mem_texture                             Counter                         # of wavefronts sent to Data-Stage from T-Stage for texture           
                                                                                                            instructions                                                          
l1tex__t_output_wavefronts_pipe_tex_realtime                                Counter                         # of wavefronts sent to Data-Stage from T-Stage for texture and       
                                                                                                            surface instructions                                                  
l1tex__t_output_wavefronts_realtime                                         Counter                         # of wavefronts sent to Data-Stage from T-Stage                       
l1tex__t_requests                                                           Counter         request         # of requests sent to T-Stage                                         
l1tex__t_requests_pipe_lsu                                                  Counter         request         # of requests sent to T-Stage from the LSU instruction pipeline       
l1tex__t_requests_pipe_lsu_mem_global_op_atom                               Counter         request         # of requests sent to T-Stage for global atomics                      
l1tex__t_requests_pipe_lsu_mem_global_op_ld                                 Counter         request         # of requests sent to T-Stage for global loads                        
l1tex__t_requests_pipe_lsu_mem_global_op_red                                Counter         request         # of requests sent to T-Stage for global reductions                   
l1tex__t_requests_pipe_lsu_mem_global_op_st                                 Counter         request         # of requests sent to T-Stage for global stores                       
l1tex__t_requests_pipe_lsu_mem_local_op_ld                                  Counter         request         # of requests sent to T-Stage for local loads                         
l1tex__t_requests_pipe_lsu_mem_local_op_st                                  Counter         request         # of requests sent to T-Stage for local stores                        
l1tex__t_requests_pipe_tex                                                  Counter         request         # of requests sent to T-Stage from the L1TEX X-Stage                  
l1tex__t_requests_pipe_tex_mem_surface                                      Counter         request         # of requests sent to T-Stage from the L1TEX X-Stage for surfaces     
l1tex__t_requests_pipe_tex_mem_surface_op_atom                              Counter         request         # of requests sent to T-Stage for surface atomics                     
l1tex__t_requests_pipe_tex_mem_surface_op_ld                                Counter         request         # of requests sent to T-Stage for surface loads                       
l1tex__t_requests_pipe_tex_mem_surface_op_red                               Counter         request         # of requests sent to T-Stage for surface reductions                  
l1tex__t_requests_pipe_tex_mem_surface_op_st                                Counter         request         # of requests sent to T-Stage for surface stores                      
l1tex__t_requests_pipe_tex_mem_texture                                      Counter         request         # of requests sent to T-Stage from the L1TEX X-Stage for textures     
l1tex__t_requests_pipe_tex_mem_texture_op_ld                                Counter         request         # of requests sent to T-Stage from the TEX pipe for TLD instructions  
l1tex__t_requests_pipe_tex_mem_texture_op_tex                               Counter         request         # of requests sent to T-Stage from the TEX pipe for TEX instructions  
l1tex__t_sector_hit_rate                                                    Ratio                           # of sector hits per sector                                           
l1tex__t_sector_pipe_lsu_hit_rate                                           Ratio                           # of sector hits for LSU pipe local/global per sector for LSU pipe    
                                                                                                            local/global                                                          
l1tex__t_sector_pipe_lsu_mem_global_op_atom_hit_rate                        Ratio                           # of sector hits for global atomics per sector for global atomics     
l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate                          Ratio                           # of sector hits for global loads per sector for global loads         
l1tex__t_sector_pipe_lsu_mem_global_op_red_hit_rate                         Ratio                           # of sector hits for global reductions per sector for global          
                                                                                                            reductions                                                            
l1tex__t_sector_pipe_lsu_mem_global_op_st_hit_rate                          Ratio                           # of sector hits for global stores per sector for global stores       
l1tex__t_sector_pipe_lsu_mem_local_op_ld_hit_rate                           Ratio                           # of sector hits for local loads per sector for local loads           
l1tex__t_sector_pipe_lsu_mem_local_op_st_hit_rate                           Ratio                           # of sector hits for local stores per sector for local stores         
l1tex__t_sector_pipe_tex_hit_rate                                           Ratio                           # of sector hits for TEX pipe per sector for TEX pipe                 
l1tex__t_sector_pipe_tex_mem_surface_hit_rate                               Ratio                           # of sector hits for surface per sector for surface                   
l1tex__t_sector_pipe_tex_mem_surface_op_atom_hit_rate                       Ratio                           # of sector hits for surface atomics per sector for surface atomics   
l1tex__t_sector_pipe_tex_mem_surface_op_ld_hit_rate                         Ratio                           # of sector hits for surface loads per sector for surface loads       
l1tex__t_sector_pipe_tex_mem_surface_op_red_hit_rate                        Ratio                           # of sector hits for surface reductions per sector for surface        
                                                                                                            reductions                                                            
l1tex__t_sector_pipe_tex_mem_surface_op_st_hit_rate                         Ratio                           # of sector hits for surface stores per sector for surface stores     
l1tex__t_sector_pipe_tex_mem_texture_hit_rate                               Ratio                           # of sector hits for texture per sector for texture                   
l1tex__t_sector_pipe_tex_mem_texture_op_ld_hit_rate                         Ratio                           # of sector hits for TLD instructions per sector for TLD instructions 
l1tex__t_sector_pipe_tex_mem_texture_op_tex_hit_rate                        Ratio                           # of sector hits for TEX instructions per sector for TEX instructions 
l1tex__t_sectors                                                            Counter         sector          # of sectors requested                                                
l1tex__t_sectors_lookup_hit                                                 Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit             
l1tex__t_sectors_lookup_miss                                                Counter         sector          # of sector requests to T-Stage that missed                           
l1tex__t_sectors_pipe_lsu                                                   Counter         sector          # of sectors requested for LSU pipe local/global                      
l1tex__t_sectors_pipe_lsu_lookup_hit                                        Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for LSU     
                                                                                                            pipe local/global                                                     
l1tex__t_sectors_pipe_lsu_lookup_miss                                       Counter         sector          # of sector requests to T-Stage that missed for LSU pipe local/global 
l1tex__t_sectors_pipe_lsu_mem_global_op_atom                                Counter         sector          # of sectors requested for global atomics                             
l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit                     Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for global  
                                                                                                            atomics                                                               
l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_miss                    Counter         sector          # of sector requests to T-Stage that missed for global atomics        
l1tex__t_sectors_pipe_lsu_mem_global_op_ld                                  Counter         sector          # of sectors requested for global loads                               
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit                       Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for global  
                                                                                                            loads                                                                 
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss                      Counter         sector          # of sector requests to T-Stage that missed for global loads          
l1tex__t_sectors_pipe_lsu_mem_global_op_red                                 Counter         sector          # of sectors requested for global reductions                          
l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit                      Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for global  
                                                                                                            reductions                                                            
l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_miss                     Counter         sector          # of sector requests to T-Stage that missed for global reductions     
l1tex__t_sectors_pipe_lsu_mem_global_op_st                                  Counter         sector          # of sectors requested for global stores                              
l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit                       Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for global  
                                                                                                            stores                                                                
l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss                      Counter         sector          # of sector requests to T-Stage that missed for global stores         
l1tex__t_sectors_pipe_lsu_mem_local_op_ld                                   Counter         sector          # of sectors requested for local loads                                
l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit                        Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for local   
                                                                                                            loads                                                                 
l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_miss                       Counter         sector          # of sector requests to T-Stage that missed for local loads           
l1tex__t_sectors_pipe_lsu_mem_local_op_st                                   Counter         sector          # of sectors requested for local stores                               
l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_hit                        Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for local   
                                                                                                            stores                                                                
l1tex__t_sectors_pipe_lsu_mem_local_op_st_lookup_miss                       Counter         sector          # of sector requests to T-Stage that missed for local stores          
l1tex__t_sectors_pipe_tex                                                   Counter         sector          # of sectors requested for TEX pipe                                   
l1tex__t_sectors_pipe_tex_lookup_hit                                        Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for TEX pipe
l1tex__t_sectors_pipe_tex_lookup_miss                                       Counter         sector          # of sector requests to T-Stage that missed for TEX pipe              
l1tex__t_sectors_pipe_tex_mem_surface                                       Counter         sector          # of sectors requested for surface                                    
l1tex__t_sectors_pipe_tex_mem_surface_lookup_hit                            Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for surface 
l1tex__t_sectors_pipe_tex_mem_surface_lookup_miss                           Counter         sector          # of sector requests to T-Stage that missed for surface               
l1tex__t_sectors_pipe_tex_mem_surface_op_atom                               Counter         sector          # of sectors requested for surface atomics                            
l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_hit                    Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for surface 
                                                                                                            atomics                                                               
l1tex__t_sectors_pipe_tex_mem_surface_op_atom_lookup_miss                   Counter         sector          # of sector requests to T-Stage that missed for surface atomics       
l1tex__t_sectors_pipe_tex_mem_surface_op_ld                                 Counter         sector          # of sectors requested for surface loads                              
l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_hit                      Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for surface 
                                                                                                            loads                                                                 
l1tex__t_sectors_pipe_tex_mem_surface_op_ld_lookup_miss                     Counter         sector          # of sector requests to T-Stage that missed for surface loads         
l1tex__t_sectors_pipe_tex_mem_surface_op_red                                Counter         sector          # of sectors requested for surface reductions                         
l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_hit                     Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for surface 
                                                                                                            reductions                                                            
l1tex__t_sectors_pipe_tex_mem_surface_op_red_lookup_miss                    Counter         sector          # of sector requests to T-Stage that missed for surface reductions    
l1tex__t_sectors_pipe_tex_mem_surface_op_st                                 Counter         sector          # of sectors requested for surface stores                             
l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_hit                      Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for surface 
                                                                                                            stores                                                                
l1tex__t_sectors_pipe_tex_mem_surface_op_st_lookup_miss                     Counter         sector          # of sector requests to T-Stage that missed for surface stores        
l1tex__t_sectors_pipe_tex_mem_texture                                       Counter         sector          # of sectors requested for texture                                    
l1tex__t_sectors_pipe_tex_mem_texture_lookup_hit                            Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for texture 
l1tex__t_sectors_pipe_tex_mem_texture_lookup_miss                           Counter         sector          # of sector requests to T-Stage that missed for texture               
l1tex__t_sectors_pipe_tex_mem_texture_op_ld                                 Counter         sector          # of sectors requested for TLD instructions                           
l1tex__t_sectors_pipe_tex_mem_texture_op_ld_lookup_hit                      Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for TLD     
                                                                                                            instructions                                                          
l1tex__t_sectors_pipe_tex_mem_texture_op_ld_lookup_miss                     Counter         sector          # of sector requests to T-Stage that missed for TLD instructions      
l1tex__t_sectors_pipe_tex_mem_texture_op_tex                                Counter         sector          # of sectors requested for TEX instructions                           
l1tex__t_sectors_pipe_tex_mem_texture_op_tex_lookup_hit                     Counter         sector          # of sector requests to T-Stage with tag-hit and data-hit for TEX     
                                                                                                            instructions                                                          
l1tex__t_sectors_pipe_tex_mem_texture_op_tex_lookup_miss                    Counter         sector          # of sector requests to T-Stage that missed for TEX instructions      
l1tex__t_set_accesses                                                       Counter                         # of cache set accesses                                               
l1tex__t_set_accesses_pipe_lsu                                              Counter                         # of cache set accesses for LSU pipe local/global                     
l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom                           Counter                         # of cache set accesses for global atomics                            
l1tex__t_set_accesses_pipe_lsu_mem_global_op_ld                             Counter                         # of cache set accesses for global loads                              
l1tex__t_set_accesses_pipe_lsu_mem_global_op_red                            Counter                         # of cache set accesses for global reductions                         
l1tex__t_set_accesses_pipe_lsu_mem_global_op_st                             Counter                         # of cache set accesses for global stores                             
l1tex__t_set_accesses_pipe_lsu_mem_local_op_ld                              Counter                         # of cache set accesses for local loads                               
l1tex__t_set_accesses_pipe_lsu_mem_local_op_st                              Counter                         # of cache set accesses for local stores                              
l1tex__t_set_accesses_pipe_tex                                              Counter                         # of cache set accesses for TEX pipe                                  
l1tex__t_set_accesses_pipe_tex_mem_surface                                  Counter                         # of cache set accesses for surface                                   
l1tex__t_set_accesses_pipe_tex_mem_surface_op_atom                          Counter                         # of cache set accesses for surface atomics                           
l1tex__t_set_accesses_pipe_tex_mem_surface_op_ld                            Counter                         # of cache set accesses for surface loads                             
l1tex__t_set_accesses_pipe_tex_mem_surface_op_red                           Counter                         # of cache set accesses for surface reductions                        
l1tex__t_set_accesses_pipe_tex_mem_surface_op_st                            Counter                         # of cache set accesses for surface stores                            
l1tex__t_set_accesses_pipe_tex_mem_texture                                  Counter                         # of cache set accesses for texture                                   
l1tex__t_set_accesses_pipe_tex_mem_texture_op_ld                            Counter                         # of cache set accesses for TLD instructions                          
l1tex__t_set_accesses_pipe_tex_mem_texture_op_tex                           Counter                         # of cache set accesses for TEX instructions                          
l1tex__t_set_conflicts                                                      Counter         cycle           # of extra cycles spent on set conflicts                              
l1tex__t_set_conflicts_pipe_lsu                                             Counter         cycle           # of extra cycles spent on set conflicts for LSU pipe local/global    
l1tex__t_set_conflicts_pipe_lsu_mem_global_op_atom                          Counter         cycle           # of extra cycles spent on set conflicts for global atomics           
l1tex__t_set_conflicts_pipe_lsu_mem_global_op_ld                            Counter         cycle           # of extra cycles spent on set conflicts for global loads             
l1tex__t_set_conflicts_pipe_lsu_mem_global_op_red                           Counter         cycle           # of extra cycles spent on set conflicts for global reductions        
l1tex__t_set_conflicts_pipe_lsu_mem_global_op_st                            Counter         cycle           # of extra cycles spent on set conflicts for global stores            
l1tex__t_set_conflicts_pipe_lsu_mem_local_op_ld                             Counter         cycle           # of extra cycles spent on set conflicts for local loads              
l1tex__t_set_conflicts_pipe_lsu_mem_local_op_st                             Counter         cycle           # of extra cycles spent on set conflicts for local stores             
l1tex__t_set_conflicts_pipe_tex                                             Counter         cycle           # of extra cycles spent on set conflicts for TEX pipe                 
l1tex__t_set_conflicts_pipe_tex_mem_surface                                 Counter         cycle           # of extra cycles spent on set conflicts for surface                  
l1tex__t_set_conflicts_pipe_tex_mem_surface_op_atom                         Counter         cycle           # of extra cycles spent on set conflicts for surface atomics          
l1tex__t_set_conflicts_pipe_tex_mem_surface_op_ld                           Counter         cycle           # of extra cycles spent on set conflicts for surface loads            
l1tex__t_set_conflicts_pipe_tex_mem_surface_op_red                          Counter         cycle           # of extra cycles spent on set conflicts for surface reductions       
l1tex__t_set_conflicts_pipe_tex_mem_surface_op_st                           Counter         cycle           # of extra cycles spent on set conflicts for surface stores           
l1tex__t_set_conflicts_pipe_tex_mem_texture                                 Counter         cycle           # of extra cycles spent on set conflicts for texture                  
l1tex__t_set_conflicts_pipe_tex_mem_texture_op_ld                           Counter         cycle           # of extra cycles spent on set conflicts for TLD instructions         
l1tex__t_set_conflicts_pipe_tex_mem_texture_op_tex                          Counter         cycle           # of extra cycles spent on set conflicts for TEX instructions         
l1tex__tex_writeback_active                                                 Counter         cycle           # of cycles where texture/surface writeback interface was active      
l1tex__texin_cycles_stalled_on_tsl1_miss                                    Counter         cycle           # of cycles where TEXIN was stalled on TSL1 cache miss, requesting    
                                                                                                            texture or sampler header                                             
l1tex__texin_requests                                                       Counter         request         # of requests sent to TEXIN                                           
l1tex__texin_requests_mem_surface                                           Counter         request         # of surface requests sent to TEXIN                                   
l1tex__texin_requests_mem_surface_op_atom                                   Counter         request         # of requests sent to TEXIN for surface atomic instructions (SUATOM,  
                                                                                                            SUATOM.CAS)                                                           
l1tex__texin_requests_mem_surface_op_ld                                     Counter         request         # of requests sent to TEXIN for surface load instructions (SULD)      
l1tex__texin_requests_mem_surface_op_null                                   Counter         request         # of requests sent to TEXIN for null surface instructions             
l1tex__texin_requests_mem_surface_op_red                                    Counter         request         # of requests sent to TEXIN for surface reduction instructions (SURED)
l1tex__texin_requests_mem_surface_op_st                                     Counter         request         # of requests sent to TEXIN for surface store instructions (SUST)     
l1tex__texin_requests_mem_texture                                           Counter         request         # of texture requests (quads) sent to TEXIN                           
l1tex__texin_requests_mem_texture_op_null                                   Counter         request         # of requests sent to TEXIN for null texture instructions             
l1tex__texin_sm2tex_req_cycles_active                                       Counter         cycle           # of cycles where interface carrying requests from SM to L1TEX        
                                                                                                            TEXIN-Stage was active                                                
l1tex__texin_sm2tex_req_cycles_stalled                                      Counter         cycle           # of cycles where interface carrying requests from SM to L1TEX        
                                                                                                            TEXIN-Stage was stalled                                               
l1tex__throughput                                                           Throughput      %               L1TEX Throughput                                                      
lts__average_gcomp_input_sector_compression_rate                            Ratio                           average compression rate achieved by all data                         
lts__average_gcomp_input_sector_success_rate                                Ratio                           average # of sectors compressed by GC                                 
lts__average_gcomp_output_sector_compression_achieved_rate                  Ratio                           average compression rate achieved by compressible data                
lts__cycles_active                                                          Counter         cycle           # of cycles where LTS was active                                      
lts__cycles_elapsed                                                         Counter         cycle           # of cycles elapsed on LTS                                            
lts__cycles_in_frame                                                        Counter         cycle           # of cycles in user-defined frame                                     
lts__cycles_in_region                                                       Counter         cycle           # of cycles in user-defined region                                    
lts__d_atomic_input_cycles_active                                           Counter         cycle           # of cycles where the atomic unit's input was active                  
lts__d_sectors                                                              Counter         sector          # of sectors accessed in data banks                                   
lts__d_sectors_fill_device                                                  Counter         sector          # of sectors filled from device memory                                
lts__d_sectors_fill_sysmem                                                  Counter         sector          # of sectors filled from system memory                                
lts__gcomp_input_sectors                                                    Counter         sector          # of sectors sent to GC                                               
lts__gcomp_output_sectors                                                   Counter         sector          # of sectors output by GC                                             
lts__ltcfabric2ltc_cycles_active                                            Counter         cycle           # of cycles where ltcfabric2ltc was active. This metric can reach     
                                                                                                            100% SOL when lts__ltcfabric2lts_cycles_active is saturated at 50%    
lts__ltcfabric2lts_cycles_active                                            Counter         cycle           # of cycles where ltcfabric2lts was active                            
lts__ltcfabric2lts_cycles_stalled                                           Counter         cycle           # of cycles where ltcfabric2lts was stalled                           
lts__lts2xbar_cycles_active                                                 Counter         cycle           # of cycles where interface from LTS to XBAR was active               
lts__t_bytes                                                                Counter         byte            # of bytes requested                                                  
lts__t_bytes_equiv_l1sectormiss                                             Counter         byte            # of bytes requested                                                  
lts__t_bytes_equiv_l1sectormiss_pipe_lsu                                    Counter         byte            # of bytes requested for LSU pipe local/global                        
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_atom                 Counter         byte            # of bytes requested for global atomics                               
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld                   Counter         byte            # of bytes requested for global loads                                 
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_red                  Counter         byte            # of bytes requested for global reductions                            
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st                   Counter         byte            # of bytes requested for global stores                                
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_ld                    Counter         byte            # of bytes requested for local loads                                  
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_st                    Counter         byte            # of bytes requested for local stores                                 
lts__t_bytes_equiv_l1sectormiss_pipe_tex                                    Counter         byte            # of bytes requested for TEX pipe                                     
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface                        Counter         byte            # of bytes requested for surface                                      
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_atom                Counter         byte            # of bytes requested for surface atomics                              
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_ld                  Counter         byte            # of bytes requested for surface loads                                
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_red                 Counter         byte            # of bytes requested for surface reductions                           
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_st                  Counter         byte            # of bytes requested for surface stores                               
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_texture                        Counter         byte            # of bytes requested for texture                                      
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_texture_op_ld                  Counter         byte            # of bytes requested for TLD instructions                             
lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_texture_op_tex                 Counter         byte            # of bytes requested for TEX instructions                             
lts__t_request_hit_rate                                                     Ratio                           proportion of L2 requests that hit                                    
lts__t_requests                                                             Counter         request         # of LTS requests                                                     
lts__t_requests_aperture_device                                             Counter         request         # of LTS requests accessing device memory (vidmem)                    
lts__t_requests_aperture_device_evict_first                                 Counter         request         # of LTS requests accessing device memory (vidmem) marked evict-first 
lts__t_requests_aperture_device_evict_first_lookup_hit                      Counter         request         # of LTS requests accessing device memory (vidmem) marked evict-first 
                                                                                                            that hit                                                              
lts__t_requests_aperture_device_evict_first_lookup_miss                     Counter         request         # of LTS requests accessing device memory (vidmem) marked evict-first 
                                                                                                            that missed                                                           
lts__t_requests_aperture_device_evict_last                                  Counter         request         # of LTS requests accessing device memory (vidmem) marked evict-last  
lts__t_requests_aperture_device_evict_last_lookup_hit                       Counter         request         # of LTS requests accessing device memory (vidmem) marked evict-last  
                                                                                                            that hit                                                              
lts__t_requests_aperture_device_evict_last_lookup_miss                      Counter         request         # of LTS requests accessing device memory (vidmem) marked evict-last  
                                                                                                            that missed                                                           
lts__t_requests_aperture_device_evict_normal                                Counter         request         # of LTS requests accessing device memory (vidmem) marked             
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_device_evict_normal_demote                         Counter         request         # of LTS requests accessing device memory (vidmem) marked             
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_device_evict_normal_lookup_hit                     Counter         request         # of LTS requests accessing device memory (vidmem) marked             
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_device_evict_normal_lookup_miss                    Counter         request         # of LTS requests accessing device memory (vidmem) marked             
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_device_lookup_hit                                  Counter         request         # of LTS requests accessing device memory (vidmem) that hit           
lts__t_requests_aperture_device_lookup_miss                                 Counter         request         # of LTS requests accessing device memory (vidmem) that missed        
lts__t_requests_aperture_device_op_atom                                     Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
lts__t_requests_aperture_device_op_atom_dot_alu                             Counter         request         # of LTS requests accessing device memory (vidmem) for atomic ALU     
                                                                                                            (non-CAS)                                                             
lts__t_requests_aperture_device_op_atom_dot_alu_lookup_hit                  Counter         request         # of LTS requests accessing device memory (vidmem) for atomic ALU     
                                                                                                            (non-CAS) that hit                                                    
lts__t_requests_aperture_device_op_atom_dot_cas                             Counter         request         # of LTS requests accessing device memory (vidmem) for atomic CAS     
lts__t_requests_aperture_device_op_atom_dot_cas_lookup_hit                  Counter         request         # of LTS requests accessing device memory (vidmem) for atomic CAS     
                                                                                                            that hit                                                              
lts__t_requests_aperture_device_op_atom_evict_first                         Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-first                                                    
lts__t_requests_aperture_device_op_atom_evict_first_lookup_hit              Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-first that hit                                           
lts__t_requests_aperture_device_op_atom_evict_first_lookup_miss             Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-first that missed                                        
lts__t_requests_aperture_device_op_atom_evict_last                          Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-last                                                     
lts__t_requests_aperture_device_op_atom_evict_last_lookup_hit               Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-last that hit                                            
lts__t_requests_aperture_device_op_atom_evict_last_lookup_miss              Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-last that missed                                         
lts__t_requests_aperture_device_op_atom_evict_normal                        Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_aperture_device_op_atom_evict_normal_lookup_hit             Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_aperture_device_op_atom_evict_normal_lookup_miss            Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_aperture_device_op_atom_lookup_hit                          Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            that hit                                                              
lts__t_requests_aperture_device_op_atom_lookup_miss                         Counter         request         # of LTS requests accessing device memory (vidmem) for all atomics    
                                                                                                            that missed                                                           
lts__t_requests_aperture_device_op_membar                                   Counter         request         # of LTS requests accessing device memory (vidmem) for memory barriers
lts__t_requests_aperture_device_op_membar_evict_first                       Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-first                                           
lts__t_requests_aperture_device_op_membar_evict_first_lookup_hit            Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-first that hit                                  
lts__t_requests_aperture_device_op_membar_evict_first_lookup_miss           Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-first that missed                               
lts__t_requests_aperture_device_op_membar_evict_last                        Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-last                                            
lts__t_requests_aperture_device_op_membar_evict_last_lookup_hit             Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-last that hit                                   
lts__t_requests_aperture_device_op_membar_evict_last_lookup_miss            Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-last that missed                                
lts__t_requests_aperture_device_op_membar_evict_normal                      Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-normal (LRU)                                    
lts__t_requests_aperture_device_op_membar_evict_normal_demote               Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-normal-demote                                   
lts__t_requests_aperture_device_op_membar_evict_normal_lookup_hit           Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-normal (LRU) that hit                           
lts__t_requests_aperture_device_op_membar_evict_normal_lookup_miss          Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers marked evict-normal (LRU) that missed                        
lts__t_requests_aperture_device_op_membar_lookup_hit                        Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers that hit                                                     
lts__t_requests_aperture_device_op_membar_lookup_miss                       Counter         request         # of LTS requests accessing device memory (vidmem) for memory         
                                                                                                            barriers that missed                                                  
lts__t_requests_aperture_device_op_read                                     Counter         request         # of LTS requests accessing device memory (vidmem) for reads          
lts__t_requests_aperture_device_op_read_evict_first                         Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-first                                                           
lts__t_requests_aperture_device_op_read_evict_first_lookup_hit              Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-first that hit                                                  
lts__t_requests_aperture_device_op_read_evict_first_lookup_miss             Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-first that missed                                               
lts__t_requests_aperture_device_op_read_evict_last                          Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-last                                                            
lts__t_requests_aperture_device_op_read_evict_last_lookup_hit               Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-last that hit                                                   
lts__t_requests_aperture_device_op_read_evict_last_lookup_miss              Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-last that missed                                                
lts__t_requests_aperture_device_op_read_evict_normal                        Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_device_op_read_evict_normal_demote                 Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_device_op_read_evict_normal_lookup_hit             Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_device_op_read_evict_normal_lookup_miss            Counter         request         # of LTS requests accessing device memory (vidmem) for reads marked   
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_device_op_read_lookup_hit                          Counter         request         # of LTS requests accessing device memory (vidmem) for reads that hit 
lts__t_requests_aperture_device_op_read_lookup_miss                         Counter         request         # of LTS requests accessing device memory (vidmem) for reads that     
                                                                                                            missed                                                                
lts__t_requests_aperture_device_op_red                                      Counter         request         # of LTS requests accessing device memory (vidmem) for reductions     
lts__t_requests_aperture_device_op_red_lookup_hit                           Counter         request         # of LTS requests accessing device memory (vidmem) for reductions     
                                                                                                            that hit                                                              
lts__t_requests_aperture_device_op_red_lookup_miss                          Counter         request         # of LTS requests accessing device memory (vidmem) for reductions     
                                                                                                            that missed                                                           
lts__t_requests_aperture_device_op_write                                    Counter         request         # of LTS requests accessing device memory (vidmem) for writes         
lts__t_requests_aperture_device_op_write_evict_first                        Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-first                                                           
lts__t_requests_aperture_device_op_write_evict_first_lookup_hit             Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-first that hit                                                  
lts__t_requests_aperture_device_op_write_evict_first_lookup_miss            Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-first that missed                                               
lts__t_requests_aperture_device_op_write_evict_last                         Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-last                                                            
lts__t_requests_aperture_device_op_write_evict_last_lookup_hit              Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-last that hit                                                   
lts__t_requests_aperture_device_op_write_evict_last_lookup_miss             Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-last that missed                                                
lts__t_requests_aperture_device_op_write_evict_normal                       Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_device_op_write_evict_normal_demote                Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_device_op_write_evict_normal_lookup_hit            Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_device_op_write_evict_normal_lookup_miss           Counter         request         # of LTS requests accessing device memory (vidmem) for writes marked  
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_device_op_write_lookup_hit                         Counter         request         # of LTS requests accessing device memory (vidmem) for writes that hit
lts__t_requests_aperture_device_op_write_lookup_miss                        Counter         request         # of LTS requests accessing device memory (vidmem) for writes that    
                                                                                                            missed                                                                
lts__t_requests_aperture_peer                                               Counter         request         # of LTS requests accessing peer memory (peermem)                     
lts__t_requests_aperture_peer_evict_first                                   Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-first  
lts__t_requests_aperture_peer_evict_first_lookup_hit                        Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-first  
                                                                                                            that hit                                                              
lts__t_requests_aperture_peer_evict_first_lookup_miss                       Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-first  
                                                                                                            that missed                                                           
lts__t_requests_aperture_peer_evict_last                                    Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-last   
lts__t_requests_aperture_peer_evict_last_lookup_hit                         Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-last   
                                                                                                            that hit                                                              
lts__t_requests_aperture_peer_evict_last_lookup_miss                        Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-last   
                                                                                                            that missed                                                           
lts__t_requests_aperture_peer_evict_normal                                  Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-normal 
                                                                                                            (LRU)                                                                 
lts__t_requests_aperture_peer_evict_normal_demote                           Counter         request         # of LTS requests accessing peer memory (peermem) marked              
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_peer_evict_normal_lookup_hit                       Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-normal 
                                                                                                            (LRU) that hit                                                        
lts__t_requests_aperture_peer_evict_normal_lookup_miss                      Counter         request         # of LTS requests accessing peer memory (peermem) marked evict-normal 
                                                                                                            (LRU) that missed                                                     
lts__t_requests_aperture_peer_lookup_hit                                    Counter         request         # of LTS requests accessing peer memory (peermem) that hit            
lts__t_requests_aperture_peer_lookup_miss                                   Counter         request         # of LTS requests accessing peer memory (peermem) that missed         
lts__t_requests_aperture_peer_op_atom                                       Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
lts__t_requests_aperture_peer_op_atom_dot_alu                               Counter         request         # of LTS requests accessing peer memory (peermem) for atomic ALU      
                                                                                                            (non-CAS)                                                             
lts__t_requests_aperture_peer_op_atom_dot_alu_lookup_hit                    Counter         request         # of LTS requests accessing peer memory (peermem) for atomic ALU      
                                                                                                            (non-CAS) that hit                                                    
lts__t_requests_aperture_peer_op_atom_dot_cas                               Counter         request         # of LTS requests accessing peer memory (peermem) for atomic CAS      
lts__t_requests_aperture_peer_op_atom_dot_cas_lookup_hit                    Counter         request         # of LTS requests accessing peer memory (peermem) for atomic CAS that 
                                                                                                            hit                                                                   
lts__t_requests_aperture_peer_op_atom_evict_first                           Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-first                                                    
lts__t_requests_aperture_peer_op_atom_evict_first_lookup_hit                Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-first that hit                                           
lts__t_requests_aperture_peer_op_atom_evict_first_lookup_miss               Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-first that missed                                        
lts__t_requests_aperture_peer_op_atom_evict_last                            Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-last                                                     
lts__t_requests_aperture_peer_op_atom_evict_last_lookup_hit                 Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-last that hit                                            
lts__t_requests_aperture_peer_op_atom_evict_last_lookup_miss                Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-last that missed                                         
lts__t_requests_aperture_peer_op_atom_evict_normal                          Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_aperture_peer_op_atom_evict_normal_lookup_hit               Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_aperture_peer_op_atom_evict_normal_lookup_miss              Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_aperture_peer_op_atom_lookup_hit                            Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            that hit                                                              
lts__t_requests_aperture_peer_op_atom_lookup_miss                           Counter         request         # of LTS requests accessing peer memory (peermem) for all atomics     
                                                                                                            that missed                                                           
lts__t_requests_aperture_peer_op_membar                                     Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
lts__t_requests_aperture_peer_op_membar_evict_first                         Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-first                                                    
lts__t_requests_aperture_peer_op_membar_evict_first_lookup_hit              Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-first that hit                                           
lts__t_requests_aperture_peer_op_membar_evict_first_lookup_miss             Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-first that missed                                        
lts__t_requests_aperture_peer_op_membar_evict_last                          Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-last                                                     
lts__t_requests_aperture_peer_op_membar_evict_last_lookup_hit               Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-last that hit                                            
lts__t_requests_aperture_peer_op_membar_evict_last_lookup_miss              Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-last that missed                                         
lts__t_requests_aperture_peer_op_membar_evict_normal                        Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_aperture_peer_op_membar_evict_normal_demote                 Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_aperture_peer_op_membar_evict_normal_lookup_hit             Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_aperture_peer_op_membar_evict_normal_lookup_miss            Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_aperture_peer_op_membar_lookup_hit                          Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            that hit                                                              
lts__t_requests_aperture_peer_op_membar_lookup_miss                         Counter         request         # of LTS requests accessing peer memory (peermem) for memory barriers 
                                                                                                            that missed                                                           
lts__t_requests_aperture_peer_op_read                                       Counter         request         # of LTS requests accessing peer memory (peermem) for reads           
lts__t_requests_aperture_peer_op_read_evict_first                           Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-first                                                           
lts__t_requests_aperture_peer_op_read_evict_first_lookup_hit                Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-first that hit                                                  
lts__t_requests_aperture_peer_op_read_evict_first_lookup_miss               Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-first that missed                                               
lts__t_requests_aperture_peer_op_read_evict_last                            Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-last                                                            
lts__t_requests_aperture_peer_op_read_evict_last_lookup_hit                 Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-last that hit                                                   
lts__t_requests_aperture_peer_op_read_evict_last_lookup_miss                Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-last that missed                                                
lts__t_requests_aperture_peer_op_read_evict_normal                          Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_peer_op_read_evict_normal_demote                   Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_peer_op_read_evict_normal_lookup_hit               Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_peer_op_read_evict_normal_lookup_miss              Counter         request         # of LTS requests accessing peer memory (peermem) for reads marked    
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_peer_op_read_lookup_hit                            Counter         request         # of LTS requests accessing peer memory (peermem) for reads that hit  
lts__t_requests_aperture_peer_op_read_lookup_miss                           Counter         request         # of LTS requests accessing peer memory (peermem) for reads that      
                                                                                                            missed                                                                
lts__t_requests_aperture_peer_op_red                                        Counter         request         # of LTS requests accessing peer memory (peermem) for reductions      
lts__t_requests_aperture_peer_op_red_lookup_hit                             Counter         request         # of LTS requests accessing peer memory (peermem) for reductions that 
                                                                                                            hit                                                                   
lts__t_requests_aperture_peer_op_red_lookup_miss                            Counter         request         # of LTS requests accessing peer memory (peermem) for reductions that 
                                                                                                            missed                                                                
lts__t_requests_aperture_peer_op_write                                      Counter         request         # of LTS requests accessing peer memory (peermem) for writes          
lts__t_requests_aperture_peer_op_write_evict_first                          Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-first                                                           
lts__t_requests_aperture_peer_op_write_evict_first_lookup_hit               Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-first that hit                                                  
lts__t_requests_aperture_peer_op_write_evict_first_lookup_miss              Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-first that missed                                               
lts__t_requests_aperture_peer_op_write_evict_last                           Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-last                                                            
lts__t_requests_aperture_peer_op_write_evict_last_lookup_hit                Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-last that hit                                                   
lts__t_requests_aperture_peer_op_write_evict_last_lookup_miss               Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-last that missed                                                
lts__t_requests_aperture_peer_op_write_evict_normal                         Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_peer_op_write_evict_normal_demote                  Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_peer_op_write_evict_normal_lookup_hit              Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_peer_op_write_evict_normal_lookup_miss             Counter         request         # of LTS requests accessing peer memory (peermem) for writes marked   
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_peer_op_write_lookup_hit                           Counter         request         # of LTS requests accessing peer memory (peermem) for writes that hit 
lts__t_requests_aperture_peer_op_write_lookup_miss                          Counter         request         # of LTS requests accessing peer memory (peermem) for writes that     
                                                                                                            missed                                                                
lts__t_requests_aperture_sysmem                                             Counter         request         # of LTS requests accessing system memory (sysmem)                    
lts__t_requests_aperture_sysmem_evict_first                                 Counter         request         # of LTS requests accessing system memory (sysmem) marked evict-first 
lts__t_requests_aperture_sysmem_evict_first_lookup_hit                      Counter         request         # of LTS requests accessing system memory (sysmem) marked evict-first 
                                                                                                            that hit                                                              
lts__t_requests_aperture_sysmem_evict_first_lookup_miss                     Counter         request         # of LTS requests accessing system memory (sysmem) marked evict-first 
                                                                                                            that missed                                                           
lts__t_requests_aperture_sysmem_evict_last                                  Counter         request         # of LTS requests accessing system memory (sysmem) marked evict-last  
lts__t_requests_aperture_sysmem_evict_last_lookup_hit                       Counter         request         # of LTS requests accessing system memory (sysmem) marked evict-last  
                                                                                                            that hit                                                              
lts__t_requests_aperture_sysmem_evict_last_lookup_miss                      Counter         request         # of LTS requests accessing system memory (sysmem) marked evict-last  
                                                                                                            that missed                                                           
lts__t_requests_aperture_sysmem_evict_normal                                Counter         request         # of LTS requests accessing system memory (sysmem) marked             
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_sysmem_evict_normal_demote                         Counter         request         # of LTS requests accessing system memory (sysmem) marked             
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_sysmem_evict_normal_lookup_hit                     Counter         request         # of LTS requests accessing system memory (sysmem) marked             
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_sysmem_evict_normal_lookup_miss                    Counter         request         # of LTS requests accessing system memory (sysmem) marked             
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_sysmem_lookup_hit                                  Counter         request         # of LTS requests accessing system memory (sysmem) that hit           
lts__t_requests_aperture_sysmem_lookup_miss                                 Counter         request         # of LTS requests accessing system memory (sysmem) that missed        
lts__t_requests_aperture_sysmem_op_atom                                     Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
lts__t_requests_aperture_sysmem_op_atom_dot_alu                             Counter         request         # of LTS requests accessing system memory (sysmem) for atomic ALU     
                                                                                                            (non-CAS)                                                             
lts__t_requests_aperture_sysmem_op_atom_dot_alu_lookup_hit                  Counter         request         # of LTS requests accessing system memory (sysmem) for atomic ALU     
                                                                                                            (non-CAS) that hit                                                    
lts__t_requests_aperture_sysmem_op_atom_dot_cas                             Counter         request         # of LTS requests accessing system memory (sysmem) for atomic CAS     
lts__t_requests_aperture_sysmem_op_atom_dot_cas_lookup_hit                  Counter         request         # of LTS requests accessing system memory (sysmem) for atomic CAS     
                                                                                                            that hit                                                              
lts__t_requests_aperture_sysmem_op_atom_evict_first                         Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-first                                                    
lts__t_requests_aperture_sysmem_op_atom_evict_first_lookup_hit              Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-first that hit                                           
lts__t_requests_aperture_sysmem_op_atom_evict_first_lookup_miss             Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-first that missed                                        
lts__t_requests_aperture_sysmem_op_atom_evict_last                          Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-last                                                     
lts__t_requests_aperture_sysmem_op_atom_evict_last_lookup_hit               Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-last that hit                                            
lts__t_requests_aperture_sysmem_op_atom_evict_last_lookup_miss              Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-last that missed                                         
lts__t_requests_aperture_sysmem_op_atom_evict_normal                        Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_aperture_sysmem_op_atom_evict_normal_lookup_hit             Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_aperture_sysmem_op_atom_evict_normal_lookup_miss            Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_aperture_sysmem_op_atom_lookup_hit                          Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            that hit                                                              
lts__t_requests_aperture_sysmem_op_atom_lookup_miss                         Counter         request         # of LTS requests accessing system memory (sysmem) for all atomics    
                                                                                                            that missed                                                           
lts__t_requests_aperture_sysmem_op_membar                                   Counter         request         # of LTS requests accessing system memory (sysmem) for memory barriers
lts__t_requests_aperture_sysmem_op_membar_evict_first                       Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-first                                           
lts__t_requests_aperture_sysmem_op_membar_evict_first_lookup_hit            Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-first that hit                                  
lts__t_requests_aperture_sysmem_op_membar_evict_first_lookup_miss           Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-first that missed                               
lts__t_requests_aperture_sysmem_op_membar_evict_last                        Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-last                                            
lts__t_requests_aperture_sysmem_op_membar_evict_last_lookup_hit             Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-last that hit                                   
lts__t_requests_aperture_sysmem_op_membar_evict_last_lookup_miss            Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-last that missed                                
lts__t_requests_aperture_sysmem_op_membar_evict_normal                      Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-normal (LRU)                                    
lts__t_requests_aperture_sysmem_op_membar_evict_normal_demote               Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-normal-demote                                   
lts__t_requests_aperture_sysmem_op_membar_evict_normal_lookup_hit           Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-normal (LRU) that hit                           
lts__t_requests_aperture_sysmem_op_membar_evict_normal_lookup_miss          Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers marked evict-normal (LRU) that missed                        
lts__t_requests_aperture_sysmem_op_membar_lookup_hit                        Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers that hit                                                     
lts__t_requests_aperture_sysmem_op_membar_lookup_miss                       Counter         request         # of LTS requests accessing system memory (sysmem) for memory         
                                                                                                            barriers that missed                                                  
lts__t_requests_aperture_sysmem_op_read                                     Counter         request         # of LTS requests accessing system memory (sysmem) for reads          
lts__t_requests_aperture_sysmem_op_read_evict_first                         Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-first                                                           
lts__t_requests_aperture_sysmem_op_read_evict_first_lookup_hit              Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-first that hit                                                  
lts__t_requests_aperture_sysmem_op_read_evict_first_lookup_miss             Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-first that missed                                               
lts__t_requests_aperture_sysmem_op_read_evict_last                          Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-last                                                            
lts__t_requests_aperture_sysmem_op_read_evict_last_lookup_hit               Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-last that hit                                                   
lts__t_requests_aperture_sysmem_op_read_evict_last_lookup_miss              Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-last that missed                                                
lts__t_requests_aperture_sysmem_op_read_evict_normal                        Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_sysmem_op_read_evict_normal_demote                 Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_sysmem_op_read_evict_normal_lookup_hit             Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_sysmem_op_read_evict_normal_lookup_miss            Counter         request         # of LTS requests accessing system memory (sysmem) for reads marked   
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_sysmem_op_read_lookup_hit                          Counter         request         # of LTS requests accessing system memory (sysmem) for reads that hit 
lts__t_requests_aperture_sysmem_op_read_lookup_miss                         Counter         request         # of LTS requests accessing system memory (sysmem) for reads that     
                                                                                                            missed                                                                
lts__t_requests_aperture_sysmem_op_red                                      Counter         request         # of LTS requests accessing system memory (sysmem) for reductions     
lts__t_requests_aperture_sysmem_op_red_lookup_hit                           Counter         request         # of LTS requests accessing system memory (sysmem) for reductions     
                                                                                                            that hit                                                              
lts__t_requests_aperture_sysmem_op_red_lookup_miss                          Counter         request         # of LTS requests accessing system memory (sysmem) for reductions     
                                                                                                            that missed                                                           
lts__t_requests_aperture_sysmem_op_write                                    Counter         request         # of LTS requests accessing system memory (sysmem) for writes         
lts__t_requests_aperture_sysmem_op_write_evict_first                        Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-first                                                           
lts__t_requests_aperture_sysmem_op_write_evict_first_lookup_hit             Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-first that hit                                                  
lts__t_requests_aperture_sysmem_op_write_evict_first_lookup_miss            Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-first that missed                                               
lts__t_requests_aperture_sysmem_op_write_evict_last                         Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-last                                                            
lts__t_requests_aperture_sysmem_op_write_evict_last_lookup_hit              Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-last that hit                                                   
lts__t_requests_aperture_sysmem_op_write_evict_last_lookup_miss             Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-last that missed                                                
lts__t_requests_aperture_sysmem_op_write_evict_normal                       Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_aperture_sysmem_op_write_evict_normal_demote                Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-normal-demote                                                   
lts__t_requests_aperture_sysmem_op_write_evict_normal_lookup_hit            Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_aperture_sysmem_op_write_evict_normal_lookup_miss           Counter         request         # of LTS requests accessing system memory (sysmem) for writes marked  
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_aperture_sysmem_op_write_lookup_hit                         Counter         request         # of LTS requests accessing system memory (sysmem) for writes that hit
lts__t_requests_aperture_sysmem_op_write_lookup_miss                        Counter         request         # of LTS requests accessing system memory (sysmem) for writes that    
                                                                                                            missed                                                                
lts__t_requests_evict_first                                                 Counter         request         # of LTS requests marked evict-first                                  
lts__t_requests_evict_first_lookup_hit                                      Counter         request         # of LTS requests marked evict-first that hit                         
lts__t_requests_evict_first_lookup_miss                                     Counter         request         # of LTS requests marked evict-first that missed                      
lts__t_requests_evict_last                                                  Counter         request         # of LTS requests marked evict-last                                   
lts__t_requests_evict_last_lookup_hit                                       Counter         request         # of LTS requests marked evict-last that hit                          
lts__t_requests_evict_last_lookup_miss                                      Counter         request         # of LTS requests marked evict-last that missed                       
lts__t_requests_evict_normal                                                Counter         request         # of LTS requests marked evict-normal (LRU)                           
lts__t_requests_evict_normal_demote                                         Counter         request         # of LTS requests marked evict-normal-demote                          
lts__t_requests_evict_normal_demote_lookup_hit                              Counter         request         # of LTS requests marked evict-normal-demote that hit                 
lts__t_requests_evict_normal_demote_lookup_miss                             Counter         request         # of LTS requests marked evict-normal-demote that missed              
lts__t_requests_evict_normal_lookup_hit                                     Counter         request         # of LTS requests marked evict-normal (LRU) that hit                  
lts__t_requests_evict_normal_lookup_miss                                    Counter         request         # of LTS requests marked evict-normal (LRU) that missed               
lts__t_requests_lookup_hit                                                  Counter         request         # of LTS requests that hit                                            
lts__t_requests_lookup_miss                                                 Counter         request         # of LTS requests that missed                                         
lts__t_requests_op_atom                                                     Counter         request         # of LTS requests for all atomics                                     
lts__t_requests_op_atom_dot_alu                                             Counter         request         # of LTS requests for atomic ALU (non-CAS)                            
lts__t_requests_op_atom_dot_alu_lookup_hit                                  Counter         request         # of LTS requests for atomic ALU (non-CAS) that hit                   
lts__t_requests_op_atom_dot_cas                                             Counter         request         # of LTS requests for atomic CAS                                      
lts__t_requests_op_atom_dot_cas_lookup_hit                                  Counter         request         # of LTS requests for atomic CAS that hit                             
lts__t_requests_op_atom_evict_first                                         Counter         request         # of LTS requests for all atomics marked evict-first                  
lts__t_requests_op_atom_evict_first_lookup_hit                              Counter         request         # of LTS requests for all atomics marked evict-first that hit         
lts__t_requests_op_atom_evict_first_lookup_miss                             Counter         request         # of LTS requests for all atomics marked evict-first that missed      
lts__t_requests_op_atom_evict_last                                          Counter         request         # of LTS requests for all atomics marked evict-last                   
lts__t_requests_op_atom_evict_last_lookup_hit                               Counter         request         # of LTS requests for all atomics marked evict-last that hit          
lts__t_requests_op_atom_evict_last_lookup_miss                              Counter         request         # of LTS requests for all atomics marked evict-last that missed       
lts__t_requests_op_atom_evict_normal                                        Counter         request         # of LTS requests for all atomics marked evict-normal (LRU)           
lts__t_requests_op_atom_evict_normal_lookup_hit                             Counter         request         # of LTS requests for all atomics marked evict-normal (LRU) that hit  
lts__t_requests_op_atom_evict_normal_lookup_miss                            Counter         request         # of LTS requests for all atomics marked evict-normal (LRU) that      
                                                                                                            missed                                                                
lts__t_requests_op_atom_lookup_hit                                          Counter         request         # of LTS requests for all atomics that hit                            
lts__t_requests_op_atom_lookup_miss                                         Counter         request         # of LTS requests for all atomics that missed                         
lts__t_requests_op_membar                                                   Counter         request         # of LTS requests for memory barriers                                 
lts__t_requests_op_membar_evict_first                                       Counter         request         # of LTS requests for memory barriers marked evict-first              
lts__t_requests_op_membar_evict_first_lookup_hit                            Counter         request         # of LTS requests for memory barriers marked evict-first that hit     
lts__t_requests_op_membar_evict_first_lookup_miss                           Counter         request         # of LTS requests for memory barriers marked evict-first that missed  
lts__t_requests_op_membar_evict_last                                        Counter         request         # of LTS requests for memory barriers marked evict-last               
lts__t_requests_op_membar_evict_last_lookup_hit                             Counter         request         # of LTS requests for memory barriers marked evict-last that hit      
lts__t_requests_op_membar_evict_last_lookup_miss                            Counter         request         # of LTS requests for memory barriers marked evict-last that missed   
lts__t_requests_op_membar_evict_normal                                      Counter         request         # of LTS requests for memory barriers marked evict-normal (LRU)       
lts__t_requests_op_membar_evict_normal_demote                               Counter         request         # of LTS requests for memory barriers marked evict-normal-demote      
lts__t_requests_op_membar_evict_normal_lookup_hit                           Counter         request         # of LTS requests for memory barriers marked evict-normal (LRU) that  
                                                                                                            hit                                                                   
lts__t_requests_op_membar_evict_normal_lookup_miss                          Counter         request         # of LTS requests for memory barriers marked evict-normal (LRU) that  
                                                                                                            missed                                                                
lts__t_requests_op_membar_lookup_hit                                        Counter         request         # of LTS requests for memory barriers that hit                        
lts__t_requests_op_membar_lookup_miss                                       Counter         request         # of LTS requests for memory barriers that missed                     
lts__t_requests_op_read                                                     Counter         request         # of LTS requests for reads                                           
lts__t_requests_op_read_evict_first                                         Counter         request         # of LTS requests for reads marked evict-first                        
lts__t_requests_op_read_evict_first_lookup_hit                              Counter         request         # of LTS requests for reads marked evict-first that hit               
lts__t_requests_op_read_evict_first_lookup_miss                             Counter         request         # of LTS requests for reads marked evict-first that missed            
lts__t_requests_op_read_evict_last                                          Counter         request         # of LTS requests for reads marked evict-last                         
lts__t_requests_op_read_evict_last_lookup_hit                               Counter         request         # of LTS requests for reads marked evict-last that hit                
lts__t_requests_op_read_evict_last_lookup_miss                              Counter         request         # of LTS requests for reads marked evict-last that missed             
lts__t_requests_op_read_evict_normal                                        Counter         request         # of LTS requests for reads marked evict-normal (LRU)                 
lts__t_requests_op_read_evict_normal_demote                                 Counter         request         # of LTS requests for reads marked evict-normal-demote                
lts__t_requests_op_read_evict_normal_lookup_hit                             Counter         request         # of LTS requests for reads marked evict-normal (LRU) that hit        
lts__t_requests_op_read_evict_normal_lookup_miss                            Counter         request         # of LTS requests for reads marked evict-normal (LRU) that missed     
lts__t_requests_op_read_lookup_hit                                          Counter         request         # of LTS requests for reads that hit                                  
lts__t_requests_op_read_lookup_miss                                         Counter         request         # of LTS requests for reads that missed                               
lts__t_requests_op_red                                                      Counter         request         # of LTS requests for reductions                                      
lts__t_requests_op_red_lookup_hit                                           Counter         request         # of LTS requests for reductions that hit                             
lts__t_requests_op_red_lookup_miss                                          Counter         request         # of LTS requests for reductions that missed                          
lts__t_requests_op_write                                                    Counter         request         # of LTS requests for writes                                          
lts__t_requests_op_write_evict_first                                        Counter         request         # of LTS requests for writes marked evict-first                       
lts__t_requests_op_write_evict_first_lookup_hit                             Counter         request         # of LTS requests for writes marked evict-first that hit              
lts__t_requests_op_write_evict_first_lookup_miss                            Counter         request         # of LTS requests for writes marked evict-first that missed           
lts__t_requests_op_write_evict_last                                         Counter         request         # of LTS requests for writes marked evict-last                        
lts__t_requests_op_write_evict_last_lookup_hit                              Counter         request         # of LTS requests for writes marked evict-last that hit               
lts__t_requests_op_write_evict_last_lookup_miss                             Counter         request         # of LTS requests for writes marked evict-last that missed            
lts__t_requests_op_write_evict_normal                                       Counter         request         # of LTS requests for writes marked evict-normal (LRU)                
lts__t_requests_op_write_evict_normal_demote                                Counter         request         # of LTS requests for writes marked evict-normal-demote               
lts__t_requests_op_write_evict_normal_lookup_hit                            Counter         request         # of LTS requests for writes marked evict-normal (LRU) that hit       
lts__t_requests_op_write_evict_normal_lookup_miss                           Counter         request         # of LTS requests for writes marked evict-normal (LRU) that missed    
lts__t_requests_op_write_lookup_hit                                         Counter         request         # of LTS requests for writes that hit                                 
lts__t_requests_op_write_lookup_miss                                        Counter         request         # of LTS requests for writes that missed                              
lts__t_requests_srcnode_gpc                                                 Counter         request         # of LTS requests from node GPC                                       
lts__t_requests_srcnode_gpc_aperture_device                                 Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
lts__t_requests_srcnode_gpc_aperture_device_evict_first                     Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-first                                                    
lts__t_requests_srcnode_gpc_aperture_device_evict_first_lookup_hit          Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcnode_gpc_aperture_device_evict_first_lookup_miss         Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcnode_gpc_aperture_device_evict_last                      Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-last                                                     
lts__t_requests_srcnode_gpc_aperture_device_evict_last_lookup_hit           Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcnode_gpc_aperture_device_evict_last_lookup_miss          Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcnode_gpc_aperture_device_evict_normal                    Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcnode_gpc_aperture_device_evict_normal_demote             Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcnode_gpc_aperture_device_evict_normal_lookup_hit         Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcnode_gpc_aperture_device_evict_normal_lookup_miss        Counter         request         # of LTS requests from node GPC accessing device memory (vidmem)      
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcnode_gpc_aperture_device_lookup_hit                      Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) that 
                                                                                                            hit                                                                   
lts__t_requests_srcnode_gpc_aperture_device_lookup_miss                     Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) that 
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_aperture_device_op_atom                         Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            all atomics                                                           
lts__t_requests_srcnode_gpc_aperture_device_op_atom_dot_alu                 Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcnode_gpc_aperture_device_op_atom_dot_alu_lookup_hit      Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcnode_gpc_aperture_device_op_atom_dot_cas                 Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            atomic CAS                                                            
lts__t_requests_srcnode_gpc_aperture_device_op_atom_dot_cas_lookup_hit      Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcnode_gpc_aperture_device_op_atom_lookup_hit              Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcnode_gpc_aperture_device_op_atom_lookup_miss             Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            all atomics that missed                                               
lts__t_requests_srcnode_gpc_aperture_device_op_membar                       Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            memory barriers                                                       
lts__t_requests_srcnode_gpc_aperture_device_op_membar_lookup_hit            Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcnode_gpc_aperture_device_op_membar_lookup_miss           Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcnode_gpc_aperture_device_op_read                         Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            reads                                                                 
lts__t_requests_srcnode_gpc_aperture_device_op_read_lookup_hit              Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            reads that hit                                                        
lts__t_requests_srcnode_gpc_aperture_device_op_read_lookup_miss             Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            reads that missed                                                     
lts__t_requests_srcnode_gpc_aperture_device_op_red                          Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            reductions                                                            
lts__t_requests_srcnode_gpc_aperture_device_op_red_lookup_hit               Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            reductions that hit                                                   
lts__t_requests_srcnode_gpc_aperture_device_op_red_lookup_miss              Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            reductions that missed                                                
lts__t_requests_srcnode_gpc_aperture_device_op_write                        Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            writes                                                                
lts__t_requests_srcnode_gpc_aperture_device_op_write_lookup_hit             Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            writes that hit                                                       
lts__t_requests_srcnode_gpc_aperture_device_op_write_lookup_miss            Counter         request         # of LTS requests from node GPC accessing device memory (vidmem) for  
                                                                                                            writes that missed                                                    
lts__t_requests_srcnode_gpc_aperture_peer                                   Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
lts__t_requests_srcnode_gpc_aperture_peer_evict_first                       Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-first                                                    
lts__t_requests_srcnode_gpc_aperture_peer_evict_first_lookup_hit            Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcnode_gpc_aperture_peer_evict_first_lookup_miss           Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcnode_gpc_aperture_peer_evict_last                        Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-last                                                     
lts__t_requests_srcnode_gpc_aperture_peer_evict_last_lookup_hit             Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcnode_gpc_aperture_peer_evict_last_lookup_miss            Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcnode_gpc_aperture_peer_evict_normal                      Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcnode_gpc_aperture_peer_evict_normal_demote               Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcnode_gpc_aperture_peer_evict_normal_lookup_hit           Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcnode_gpc_aperture_peer_evict_normal_lookup_miss          Counter         request         # of LTS requests from node GPC accessing peer memory (peermem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcnode_gpc_aperture_peer_lookup_hit                        Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) that  
                                                                                                            hit                                                                   
lts__t_requests_srcnode_gpc_aperture_peer_lookup_miss                       Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) that  
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_aperture_peer_op_atom                           Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            all atomics                                                           
lts__t_requests_srcnode_gpc_aperture_peer_op_atom_dot_alu                   Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcnode_gpc_aperture_peer_op_atom_dot_alu_lookup_hit        Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcnode_gpc_aperture_peer_op_atom_dot_cas                   Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            atomic CAS                                                            
lts__t_requests_srcnode_gpc_aperture_peer_op_atom_dot_cas_lookup_hit        Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcnode_gpc_aperture_peer_op_atom_lookup_hit                Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcnode_gpc_aperture_peer_op_atom_lookup_miss               Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            all atomics that missed                                               
lts__t_requests_srcnode_gpc_aperture_peer_op_membar                         Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            memory barriers                                                       
lts__t_requests_srcnode_gpc_aperture_peer_op_membar_lookup_hit              Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcnode_gpc_aperture_peer_op_membar_lookup_miss             Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcnode_gpc_aperture_peer_op_read                           Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            reads                                                                 
lts__t_requests_srcnode_gpc_aperture_peer_op_read_lookup_hit                Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            reads that hit                                                        
lts__t_requests_srcnode_gpc_aperture_peer_op_read_lookup_miss               Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            reads that missed                                                     
lts__t_requests_srcnode_gpc_aperture_peer_op_red                            Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            reductions                                                            
lts__t_requests_srcnode_gpc_aperture_peer_op_red_lookup_hit                 Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            reductions that hit                                                   
lts__t_requests_srcnode_gpc_aperture_peer_op_red_lookup_miss                Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            reductions that missed                                                
lts__t_requests_srcnode_gpc_aperture_peer_op_write                          Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            writes                                                                
lts__t_requests_srcnode_gpc_aperture_peer_op_write_lookup_hit               Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            writes that hit                                                       
lts__t_requests_srcnode_gpc_aperture_peer_op_write_lookup_miss              Counter         request         # of LTS requests from node GPC accessing peer memory (peermem) for   
                                                                                                            writes that missed                                                    
lts__t_requests_srcnode_gpc_aperture_sysmem                                 Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_first                     Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-first                                                    
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_first_lookup_hit          Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_first_lookup_miss         Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_last                      Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-last                                                     
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_last_lookup_hit           Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_last_lookup_miss          Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_normal                    Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_normal_demote             Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_normal_lookup_hit         Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcnode_gpc_aperture_sysmem_evict_normal_lookup_miss        Counter         request         # of LTS requests from node GPC accessing system memory (sysmem)      
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcnode_gpc_aperture_sysmem_lookup_hit                      Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) that 
                                                                                                            hit                                                                   
lts__t_requests_srcnode_gpc_aperture_sysmem_lookup_miss                     Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) that 
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom                         Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            all atomics                                                           
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom_dot_alu                 Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom_dot_alu_lookup_hit      Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom_dot_cas                 Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            atomic CAS                                                            
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom_dot_cas_lookup_hit      Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom_lookup_hit              Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcnode_gpc_aperture_sysmem_op_atom_lookup_miss             Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            all atomics that missed                                               
lts__t_requests_srcnode_gpc_aperture_sysmem_op_membar                       Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            memory barriers                                                       
lts__t_requests_srcnode_gpc_aperture_sysmem_op_membar_lookup_hit            Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcnode_gpc_aperture_sysmem_op_membar_lookup_miss           Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcnode_gpc_aperture_sysmem_op_read                         Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            reads                                                                 
lts__t_requests_srcnode_gpc_aperture_sysmem_op_read_lookup_hit              Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            reads that hit                                                        
lts__t_requests_srcnode_gpc_aperture_sysmem_op_read_lookup_miss             Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            reads that missed                                                     
lts__t_requests_srcnode_gpc_aperture_sysmem_op_red                          Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            reductions                                                            
lts__t_requests_srcnode_gpc_aperture_sysmem_op_red_lookup_hit               Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            reductions that hit                                                   
lts__t_requests_srcnode_gpc_aperture_sysmem_op_red_lookup_miss              Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            reductions that missed                                                
lts__t_requests_srcnode_gpc_aperture_sysmem_op_write                        Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            writes                                                                
lts__t_requests_srcnode_gpc_aperture_sysmem_op_write_lookup_hit             Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            writes that hit                                                       
lts__t_requests_srcnode_gpc_aperture_sysmem_op_write_lookup_miss            Counter         request         # of LTS requests from node GPC accessing system memory (sysmem) for  
                                                                                                            writes that missed                                                    
lts__t_requests_srcnode_gpc_evict_first                                     Counter         request         # of LTS requests from node GPC marked evict-first                    
lts__t_requests_srcnode_gpc_evict_first_lookup_hit                          Counter         request         # of LTS requests from node GPC marked evict-first that hit           
lts__t_requests_srcnode_gpc_evict_first_lookup_miss                         Counter         request         # of LTS requests from node GPC marked evict-first that missed        
lts__t_requests_srcnode_gpc_evict_last                                      Counter         request         # of LTS requests from node GPC marked evict-last                     
lts__t_requests_srcnode_gpc_evict_last_lookup_hit                           Counter         request         # of LTS requests from node GPC marked evict-last that hit            
lts__t_requests_srcnode_gpc_evict_last_lookup_miss                          Counter         request         # of LTS requests from node GPC marked evict-last that missed         
lts__t_requests_srcnode_gpc_evict_normal                                    Counter         request         # of LTS requests from node GPC marked evict-normal (LRU)             
lts__t_requests_srcnode_gpc_evict_normal_demote                             Counter         request         # of LTS requests from node GPC marked evict-normal-demote            
lts__t_requests_srcnode_gpc_evict_normal_demote_lookup_hit                  Counter         request         # of LTS requests from node GPC marked evict-normal-demote that hit   
lts__t_requests_srcnode_gpc_evict_normal_demote_lookup_miss                 Counter         request         # of LTS requests from node GPC marked evict-normal-demote that missed
lts__t_requests_srcnode_gpc_evict_normal_lookup_hit                         Counter         request         # of LTS requests from node GPC marked evict-normal (LRU) that hit    
lts__t_requests_srcnode_gpc_evict_normal_lookup_miss                        Counter         request         # of LTS requests from node GPC marked evict-normal (LRU) that missed 
lts__t_requests_srcnode_gpc_lookup_hit                                      Counter         request         # of LTS requests from node GPC that hit                              
lts__t_requests_srcnode_gpc_lookup_miss                                     Counter         request         # of LTS requests from node GPC that missed                           
lts__t_requests_srcnode_gpc_op_atom                                         Counter         request         # of LTS requests from node GPC for all atomics                       
lts__t_requests_srcnode_gpc_op_atom_dot_alu                                 Counter         request         # of LTS requests from node GPC for atomic ALU (non-CAS)              
lts__t_requests_srcnode_gpc_op_atom_dot_alu_lookup_hit                      Counter         request         # of LTS requests from node GPC for atomic ALU (non-CAS) that hit     
lts__t_requests_srcnode_gpc_op_atom_dot_cas                                 Counter         request         # of LTS requests from node GPC for atomic CAS                        
lts__t_requests_srcnode_gpc_op_atom_dot_cas_lookup_hit                      Counter         request         # of LTS requests from node GPC for atomic CAS that hit               
lts__t_requests_srcnode_gpc_op_atom_evict_first                             Counter         request         # of LTS requests from node GPC for all atomics marked evict-first    
lts__t_requests_srcnode_gpc_op_atom_evict_first_lookup_hit                  Counter         request         # of LTS requests from node GPC for all atomics marked evict-first    
                                                                                                            that hit                                                              
lts__t_requests_srcnode_gpc_op_atom_evict_first_lookup_miss                 Counter         request         # of LTS requests from node GPC for all atomics marked evict-first    
                                                                                                            that missed                                                           
lts__t_requests_srcnode_gpc_op_atom_evict_last                              Counter         request         # of LTS requests from node GPC for all atomics marked evict-last     
lts__t_requests_srcnode_gpc_op_atom_evict_last_lookup_hit                   Counter         request         # of LTS requests from node GPC for all atomics marked evict-last     
                                                                                                            that hit                                                              
lts__t_requests_srcnode_gpc_op_atom_evict_last_lookup_miss                  Counter         request         # of LTS requests from node GPC for all atomics marked evict-last     
                                                                                                            that missed                                                           
lts__t_requests_srcnode_gpc_op_atom_evict_normal                            Counter         request         # of LTS requests from node GPC for all atomics marked evict-normal   
                                                                                                            (LRU)                                                                 
lts__t_requests_srcnode_gpc_op_atom_evict_normal_lookup_hit                 Counter         request         # of LTS requests from node GPC for all atomics marked evict-normal   
                                                                                                            (LRU) that hit                                                        
lts__t_requests_srcnode_gpc_op_atom_evict_normal_lookup_miss                Counter         request         # of LTS requests from node GPC for all atomics marked evict-normal   
                                                                                                            (LRU) that missed                                                     
lts__t_requests_srcnode_gpc_op_atom_lookup_hit                              Counter         request         # of LTS requests from node GPC for all atomics that hit              
lts__t_requests_srcnode_gpc_op_atom_lookup_miss                             Counter         request         # of LTS requests from node GPC for all atomics that missed           
lts__t_requests_srcnode_gpc_op_membar                                       Counter         request         # of LTS requests from node GPC for memory barriers                   
lts__t_requests_srcnode_gpc_op_membar_evict_first                           Counter         request         # of LTS requests from node GPC for memory barriers marked evict-first
lts__t_requests_srcnode_gpc_op_membar_evict_first_lookup_hit                Counter         request         # of LTS requests from node GPC for memory barriers marked            
                                                                                                            evict-first that hit                                                  
lts__t_requests_srcnode_gpc_op_membar_evict_first_lookup_miss               Counter         request         # of LTS requests from node GPC for memory barriers marked            
                                                                                                            evict-first that missed                                               
lts__t_requests_srcnode_gpc_op_membar_evict_last                            Counter         request         # of LTS requests from node GPC for memory barriers marked evict-last 
lts__t_requests_srcnode_gpc_op_membar_evict_last_lookup_hit                 Counter         request         # of LTS requests from node GPC for memory barriers marked evict-last 
                                                                                                            that hit                                                              
lts__t_requests_srcnode_gpc_op_membar_evict_last_lookup_miss                Counter         request         # of LTS requests from node GPC for memory barriers marked evict-last 
                                                                                                            that missed                                                           
lts__t_requests_srcnode_gpc_op_membar_evict_normal                          Counter         request         # of LTS requests from node GPC for memory barriers marked            
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_srcnode_gpc_op_membar_evict_normal_demote                   Counter         request         # of LTS requests from node GPC for memory barriers marked            
                                                                                                            evict-normal-demote                                                   
lts__t_requests_srcnode_gpc_op_membar_evict_normal_lookup_hit               Counter         request         # of LTS requests from node GPC for memory barriers marked            
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_srcnode_gpc_op_membar_evict_normal_lookup_miss              Counter         request         # of LTS requests from node GPC for memory barriers marked            
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_srcnode_gpc_op_membar_lookup_hit                            Counter         request         # of LTS requests from node GPC for memory barriers that hit          
lts__t_requests_srcnode_gpc_op_membar_lookup_miss                           Counter         request         # of LTS requests from node GPC for memory barriers that missed       
lts__t_requests_srcnode_gpc_op_read                                         Counter         request         # of LTS requests from node GPC for reads                             
lts__t_requests_srcnode_gpc_op_read_evict_first                             Counter         request         # of LTS requests from node GPC for reads marked evict-first          
lts__t_requests_srcnode_gpc_op_read_evict_first_lookup_hit                  Counter         request         # of LTS requests from node GPC for reads marked evict-first that hit 
lts__t_requests_srcnode_gpc_op_read_evict_first_lookup_miss                 Counter         request         # of LTS requests from node GPC for reads marked evict-first that     
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_op_read_evict_last                              Counter         request         # of LTS requests from node GPC for reads marked evict-last           
lts__t_requests_srcnode_gpc_op_read_evict_last_lookup_hit                   Counter         request         # of LTS requests from node GPC for reads marked evict-last that hit  
lts__t_requests_srcnode_gpc_op_read_evict_last_lookup_miss                  Counter         request         # of LTS requests from node GPC for reads marked evict-last that      
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_op_read_evict_normal                            Counter         request         # of LTS requests from node GPC for reads marked evict-normal (LRU)   
lts__t_requests_srcnode_gpc_op_read_evict_normal_demote                     Counter         request         # of LTS requests from node GPC for reads marked evict-normal-demote  
lts__t_requests_srcnode_gpc_op_read_evict_normal_lookup_hit                 Counter         request         # of LTS requests from node GPC for reads marked evict-normal (LRU)   
                                                                                                            that hit                                                              
lts__t_requests_srcnode_gpc_op_read_evict_normal_lookup_miss                Counter         request         # of LTS requests from node GPC for reads marked evict-normal (LRU)   
                                                                                                            that missed                                                           
lts__t_requests_srcnode_gpc_op_read_lookup_hit                              Counter         request         # of LTS requests from node GPC for reads that hit                    
lts__t_requests_srcnode_gpc_op_read_lookup_miss                             Counter         request         # of LTS requests from node GPC for reads that missed                 
lts__t_requests_srcnode_gpc_op_red                                          Counter         request         # of LTS requests from node GPC for reductions                        
lts__t_requests_srcnode_gpc_op_red_lookup_hit                               Counter         request         # of LTS requests from node GPC for reductions that hit               
lts__t_requests_srcnode_gpc_op_red_lookup_miss                              Counter         request         # of LTS requests from node GPC for reductions that missed            
lts__t_requests_srcnode_gpc_op_write                                        Counter         request         # of LTS requests from node GPC for writes                            
lts__t_requests_srcnode_gpc_op_write_evict_first                            Counter         request         # of LTS requests from node GPC for writes marked evict-first         
lts__t_requests_srcnode_gpc_op_write_evict_first_lookup_hit                 Counter         request         # of LTS requests from node GPC for writes marked evict-first that hit
lts__t_requests_srcnode_gpc_op_write_evict_first_lookup_miss                Counter         request         # of LTS requests from node GPC for writes marked evict-first that    
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_op_write_evict_last                             Counter         request         # of LTS requests from node GPC for writes marked evict-last          
lts__t_requests_srcnode_gpc_op_write_evict_last_lookup_hit                  Counter         request         # of LTS requests from node GPC for writes marked evict-last that hit 
lts__t_requests_srcnode_gpc_op_write_evict_last_lookup_miss                 Counter         request         # of LTS requests from node GPC for writes marked evict-last that     
                                                                                                            missed                                                                
lts__t_requests_srcnode_gpc_op_write_evict_normal                           Counter         request         # of LTS requests from node GPC for writes marked evict-normal (LRU)  
lts__t_requests_srcnode_gpc_op_write_evict_normal_demote                    Counter         request         # of LTS requests from node GPC for writes marked evict-normal-demote 
lts__t_requests_srcnode_gpc_op_write_evict_normal_lookup_hit                Counter         request         # of LTS requests from node GPC for writes marked evict-normal (LRU)  
                                                                                                            that hit                                                              
lts__t_requests_srcnode_gpc_op_write_evict_normal_lookup_miss               Counter         request         # of LTS requests from node GPC for writes marked evict-normal (LRU)  
                                                                                                            that missed                                                           
lts__t_requests_srcnode_gpc_op_write_lookup_hit                             Counter         request         # of LTS requests from node GPC for writes that hit                   
lts__t_requests_srcnode_gpc_op_write_lookup_miss                            Counter         request         # of LTS requests from node GPC for writes that missed                
lts__t_requests_srcunit_l1                                                  Counter         request         # of LTS requests from unit L1                                        
lts__t_requests_srcunit_l1_aperture_device                                  Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
lts__t_requests_srcunit_l1_aperture_device_evict_first                      Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_l1_aperture_device_evict_first_lookup_hit           Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcunit_l1_aperture_device_evict_first_lookup_miss          Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcunit_l1_aperture_device_evict_last                       Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_l1_aperture_device_evict_last_lookup_hit            Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcunit_l1_aperture_device_evict_last_lookup_miss           Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcunit_l1_aperture_device_evict_normal                     Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_l1_aperture_device_evict_normal_demote              Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_l1_aperture_device_evict_normal_lookup_hit          Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcunit_l1_aperture_device_evict_normal_lookup_miss         Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcunit_l1_aperture_device_lookup_hit                       Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) that  
                                                                                                            hit                                                                   
lts__t_requests_srcunit_l1_aperture_device_lookup_miss                      Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) that  
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_aperture_device_op_atom                          Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            all atomics                                                           
lts__t_requests_srcunit_l1_aperture_device_op_atom_dot_alu                  Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcunit_l1_aperture_device_op_atom_dot_alu_lookup_hit       Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcunit_l1_aperture_device_op_atom_dot_cas                  Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            atomic CAS                                                            
lts__t_requests_srcunit_l1_aperture_device_op_atom_dot_cas_lookup_hit       Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcunit_l1_aperture_device_op_atom_lookup_hit               Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcunit_l1_aperture_device_op_atom_lookup_miss              Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            all atomics that missed                                               
lts__t_requests_srcunit_l1_aperture_device_op_membar                        Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_l1_aperture_device_op_membar_lookup_hit             Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcunit_l1_aperture_device_op_membar_lookup_miss            Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcunit_l1_aperture_device_op_read                          Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            reads                                                                 
lts__t_requests_srcunit_l1_aperture_device_op_read_lookup_hit               Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            reads that hit                                                        
lts__t_requests_srcunit_l1_aperture_device_op_read_lookup_miss              Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            reads that missed                                                     
lts__t_requests_srcunit_l1_aperture_device_op_red                           Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            reductions                                                            
lts__t_requests_srcunit_l1_aperture_device_op_red_lookup_hit                Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            reductions that hit                                                   
lts__t_requests_srcunit_l1_aperture_device_op_red_lookup_miss               Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            reductions that missed                                                
lts__t_requests_srcunit_l1_aperture_device_op_write                         Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            writes                                                                
lts__t_requests_srcunit_l1_aperture_device_op_write_lookup_hit              Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            writes that hit                                                       
lts__t_requests_srcunit_l1_aperture_device_op_write_lookup_miss             Counter         request         # of LTS requests from unit L1 accessing device memory (vidmem) for   
                                                                                                            writes that missed                                                    
lts__t_requests_srcunit_l1_aperture_peer                                    Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem)        
lts__t_requests_srcunit_l1_aperture_peer_evict_first                        Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-first                                                           
lts__t_requests_srcunit_l1_aperture_peer_evict_first_lookup_hit             Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-first that hit                                                  
lts__t_requests_srcunit_l1_aperture_peer_evict_first_lookup_miss            Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-first that missed                                               
lts__t_requests_srcunit_l1_aperture_peer_evict_last                         Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-last                                                            
lts__t_requests_srcunit_l1_aperture_peer_evict_last_lookup_hit              Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-last that hit                                                   
lts__t_requests_srcunit_l1_aperture_peer_evict_last_lookup_miss             Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-last that missed                                                
lts__t_requests_srcunit_l1_aperture_peer_evict_normal                       Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_srcunit_l1_aperture_peer_evict_normal_demote                Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote                                                   
lts__t_requests_srcunit_l1_aperture_peer_evict_normal_lookup_hit            Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_srcunit_l1_aperture_peer_evict_normal_lookup_miss           Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_srcunit_l1_aperture_peer_lookup_hit                         Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) that   
                                                                                                            hit                                                                   
lts__t_requests_srcunit_l1_aperture_peer_lookup_miss                        Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) that   
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_aperture_peer_op_atom                            Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            all atomics                                                           
lts__t_requests_srcunit_l1_aperture_peer_op_atom_dot_alu                    Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcunit_l1_aperture_peer_op_atom_dot_alu_lookup_hit         Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcunit_l1_aperture_peer_op_atom_dot_cas                    Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            atomic CAS                                                            
lts__t_requests_srcunit_l1_aperture_peer_op_atom_dot_cas_lookup_hit         Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcunit_l1_aperture_peer_op_atom_lookup_hit                 Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcunit_l1_aperture_peer_op_atom_lookup_miss                Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            all atomics that missed                                               
lts__t_requests_srcunit_l1_aperture_peer_op_membar                          Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_l1_aperture_peer_op_membar_lookup_hit               Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcunit_l1_aperture_peer_op_membar_lookup_miss              Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcunit_l1_aperture_peer_op_read                            Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            reads                                                                 
lts__t_requests_srcunit_l1_aperture_peer_op_read_lookup_hit                 Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            reads that hit                                                        
lts__t_requests_srcunit_l1_aperture_peer_op_read_lookup_miss                Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            reads that missed                                                     
lts__t_requests_srcunit_l1_aperture_peer_op_red                             Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            reductions                                                            
lts__t_requests_srcunit_l1_aperture_peer_op_red_lookup_hit                  Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            reductions that hit                                                   
lts__t_requests_srcunit_l1_aperture_peer_op_red_lookup_miss                 Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            reductions that missed                                                
lts__t_requests_srcunit_l1_aperture_peer_op_write                           Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            writes                                                                
lts__t_requests_srcunit_l1_aperture_peer_op_write_lookup_hit                Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            writes that hit                                                       
lts__t_requests_srcunit_l1_aperture_peer_op_write_lookup_miss               Counter         request         # of LTS requests from unit L1 accessing peer memory (peermem) for    
                                                                                                            writes that missed                                                    
lts__t_requests_srcunit_l1_aperture_sysmem                                  Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
lts__t_requests_srcunit_l1_aperture_sysmem_evict_first                      Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_l1_aperture_sysmem_evict_first_lookup_hit           Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcunit_l1_aperture_sysmem_evict_first_lookup_miss          Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcunit_l1_aperture_sysmem_evict_last                       Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_l1_aperture_sysmem_evict_last_lookup_hit            Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcunit_l1_aperture_sysmem_evict_last_lookup_miss           Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcunit_l1_aperture_sysmem_evict_normal                     Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_l1_aperture_sysmem_evict_normal_demote              Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_l1_aperture_sysmem_evict_normal_lookup_hit          Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcunit_l1_aperture_sysmem_evict_normal_lookup_miss         Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcunit_l1_aperture_sysmem_lookup_hit                       Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) that  
                                                                                                            hit                                                                   
lts__t_requests_srcunit_l1_aperture_sysmem_lookup_miss                      Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) that  
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom                          Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            all atomics                                                           
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom_dot_alu                  Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom_dot_alu_lookup_hit       Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom_dot_cas                  Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            atomic CAS                                                            
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom_dot_cas_lookup_hit       Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom_lookup_hit               Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcunit_l1_aperture_sysmem_op_atom_lookup_miss              Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            all atomics that missed                                               
lts__t_requests_srcunit_l1_aperture_sysmem_op_membar                        Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_l1_aperture_sysmem_op_membar_lookup_hit             Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcunit_l1_aperture_sysmem_op_membar_lookup_miss            Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcunit_l1_aperture_sysmem_op_read                          Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            reads                                                                 
lts__t_requests_srcunit_l1_aperture_sysmem_op_read_lookup_hit               Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            reads that hit                                                        
lts__t_requests_srcunit_l1_aperture_sysmem_op_read_lookup_miss              Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            reads that missed                                                     
lts__t_requests_srcunit_l1_aperture_sysmem_op_red                           Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            reductions                                                            
lts__t_requests_srcunit_l1_aperture_sysmem_op_red_lookup_hit                Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            reductions that hit                                                   
lts__t_requests_srcunit_l1_aperture_sysmem_op_red_lookup_miss               Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            reductions that missed                                                
lts__t_requests_srcunit_l1_aperture_sysmem_op_write                         Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            writes                                                                
lts__t_requests_srcunit_l1_aperture_sysmem_op_write_lookup_hit              Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            writes that hit                                                       
lts__t_requests_srcunit_l1_aperture_sysmem_op_write_lookup_miss             Counter         request         # of LTS requests from unit L1 accessing system memory (sysmem) for   
                                                                                                            writes that missed                                                    
lts__t_requests_srcunit_l1_evict_first                                      Counter         request         # of LTS requests from unit L1 marked evict-first                     
lts__t_requests_srcunit_l1_evict_first_lookup_hit                           Counter         request         # of LTS requests from unit L1 marked evict-first that hit            
lts__t_requests_srcunit_l1_evict_first_lookup_miss                          Counter         request         # of LTS requests from unit L1 marked evict-first that missed         
lts__t_requests_srcunit_l1_evict_last                                       Counter         request         # of LTS requests from unit L1 marked evict-last                      
lts__t_requests_srcunit_l1_evict_last_lookup_hit                            Counter         request         # of LTS requests from unit L1 marked evict-last that hit             
lts__t_requests_srcunit_l1_evict_last_lookup_miss                           Counter         request         # of LTS requests from unit L1 marked evict-last that missed          
lts__t_requests_srcunit_l1_evict_normal                                     Counter         request         # of LTS requests from unit L1 marked evict-normal (LRU)              
lts__t_requests_srcunit_l1_evict_normal_demote                              Counter         request         # of LTS requests from unit L1 marked evict-normal-demote             
lts__t_requests_srcunit_l1_evict_normal_demote_lookup_hit                   Counter         request         # of LTS requests from unit L1 marked evict-normal-demote that hit    
lts__t_requests_srcunit_l1_evict_normal_demote_lookup_miss                  Counter         request         # of LTS requests from unit L1 marked evict-normal-demote that missed 
lts__t_requests_srcunit_l1_evict_normal_lookup_hit                          Counter         request         # of LTS requests from unit L1 marked evict-normal (LRU) that hit     
lts__t_requests_srcunit_l1_evict_normal_lookup_miss                         Counter         request         # of LTS requests from unit L1 marked evict-normal (LRU) that missed  
lts__t_requests_srcunit_l1_lookup_hit                                       Counter         request         # of LTS requests from unit L1 that hit                               
lts__t_requests_srcunit_l1_lookup_miss                                      Counter         request         # of LTS requests from unit L1 that missed                            
lts__t_requests_srcunit_l1_op_atom                                          Counter         request         # of LTS requests from unit L1 for all atomics                        
lts__t_requests_srcunit_l1_op_atom_dot_alu                                  Counter         request         # of LTS requests from unit L1 for atomic ALU (non-CAS)               
lts__t_requests_srcunit_l1_op_atom_dot_alu_lookup_hit                       Counter         request         # of LTS requests from unit L1 for atomic ALU (non-CAS) that hit      
lts__t_requests_srcunit_l1_op_atom_dot_cas                                  Counter         request         # of LTS requests from unit L1 for atomic CAS                         
lts__t_requests_srcunit_l1_op_atom_dot_cas_lookup_hit                       Counter         request         # of LTS requests from unit L1 for atomic CAS that hit                
lts__t_requests_srcunit_l1_op_atom_evict_first                              Counter         request         # of LTS requests from unit L1 for all atomics marked evict-first     
lts__t_requests_srcunit_l1_op_atom_evict_first_lookup_hit                   Counter         request         # of LTS requests from unit L1 for all atomics marked evict-first     
                                                                                                            that hit                                                              
lts__t_requests_srcunit_l1_op_atom_evict_first_lookup_miss                  Counter         request         # of LTS requests from unit L1 for all atomics marked evict-first     
                                                                                                            that missed                                                           
lts__t_requests_srcunit_l1_op_atom_evict_last                               Counter         request         # of LTS requests from unit L1 for all atomics marked evict-last      
lts__t_requests_srcunit_l1_op_atom_evict_last_lookup_hit                    Counter         request         # of LTS requests from unit L1 for all atomics marked evict-last that 
                                                                                                            hit                                                                   
lts__t_requests_srcunit_l1_op_atom_evict_last_lookup_miss                   Counter         request         # of LTS requests from unit L1 for all atomics marked evict-last that 
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_op_atom_evict_normal                             Counter         request         # of LTS requests from unit L1 for all atomics marked evict-normal    
                                                                                                            (LRU)                                                                 
lts__t_requests_srcunit_l1_op_atom_evict_normal_lookup_hit                  Counter         request         # of LTS requests from unit L1 for all atomics marked evict-normal    
                                                                                                            (LRU) that hit                                                        
lts__t_requests_srcunit_l1_op_atom_evict_normal_lookup_miss                 Counter         request         # of LTS requests from unit L1 for all atomics marked evict-normal    
                                                                                                            (LRU) that missed                                                     
lts__t_requests_srcunit_l1_op_atom_lookup_hit                               Counter         request         # of LTS requests from unit L1 for all atomics that hit               
lts__t_requests_srcunit_l1_op_atom_lookup_miss                              Counter         request         # of LTS requests from unit L1 for all atomics that missed            
lts__t_requests_srcunit_l1_op_membar                                        Counter         request         # of LTS requests from unit L1 for memory barriers                    
lts__t_requests_srcunit_l1_op_membar_evict_first                            Counter         request         # of LTS requests from unit L1 for memory barriers marked evict-first 
lts__t_requests_srcunit_l1_op_membar_evict_first_lookup_hit                 Counter         request         # of LTS requests from unit L1 for memory barriers marked evict-first 
                                                                                                            that hit                                                              
lts__t_requests_srcunit_l1_op_membar_evict_first_lookup_miss                Counter         request         # of LTS requests from unit L1 for memory barriers marked evict-first 
                                                                                                            that missed                                                           
lts__t_requests_srcunit_l1_op_membar_evict_last                             Counter         request         # of LTS requests from unit L1 for memory barriers marked evict-last  
lts__t_requests_srcunit_l1_op_membar_evict_last_lookup_hit                  Counter         request         # of LTS requests from unit L1 for memory barriers marked evict-last  
                                                                                                            that hit                                                              
lts__t_requests_srcunit_l1_op_membar_evict_last_lookup_miss                 Counter         request         # of LTS requests from unit L1 for memory barriers marked evict-last  
                                                                                                            that missed                                                           
lts__t_requests_srcunit_l1_op_membar_evict_normal                           Counter         request         # of LTS requests from unit L1 for memory barriers marked             
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_srcunit_l1_op_membar_evict_normal_demote                    Counter         request         # of LTS requests from unit L1 for memory barriers marked             
                                                                                                            evict-normal-demote                                                   
lts__t_requests_srcunit_l1_op_membar_evict_normal_lookup_hit                Counter         request         # of LTS requests from unit L1 for memory barriers marked             
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_srcunit_l1_op_membar_evict_normal_lookup_miss               Counter         request         # of LTS requests from unit L1 for memory barriers marked             
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_srcunit_l1_op_membar_lookup_hit                             Counter         request         # of LTS requests from unit L1 for memory barriers that hit           
lts__t_requests_srcunit_l1_op_membar_lookup_miss                            Counter         request         # of LTS requests from unit L1 for memory barriers that missed        
lts__t_requests_srcunit_l1_op_read                                          Counter         request         # of LTS requests from unit L1 for reads                              
lts__t_requests_srcunit_l1_op_read_evict_first                              Counter         request         # of LTS requests from unit L1 for reads marked evict-first           
lts__t_requests_srcunit_l1_op_read_evict_first_lookup_hit                   Counter         request         # of LTS requests from unit L1 for reads marked evict-first that hit  
lts__t_requests_srcunit_l1_op_read_evict_first_lookup_miss                  Counter         request         # of LTS requests from unit L1 for reads marked evict-first that      
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_op_read_evict_last                               Counter         request         # of LTS requests from unit L1 for reads marked evict-last            
lts__t_requests_srcunit_l1_op_read_evict_last_lookup_hit                    Counter         request         # of LTS requests from unit L1 for reads marked evict-last that hit   
lts__t_requests_srcunit_l1_op_read_evict_last_lookup_miss                   Counter         request         # of LTS requests from unit L1 for reads marked evict-last that missed
lts__t_requests_srcunit_l1_op_read_evict_normal                             Counter         request         # of LTS requests from unit L1 for reads marked evict-normal (LRU)    
lts__t_requests_srcunit_l1_op_read_evict_normal_demote                      Counter         request         # of LTS requests from unit L1 for reads marked evict-normal-demote   
lts__t_requests_srcunit_l1_op_read_evict_normal_lookup_hit                  Counter         request         # of LTS requests from unit L1 for reads marked evict-normal (LRU)    
                                                                                                            that hit                                                              
lts__t_requests_srcunit_l1_op_read_evict_normal_lookup_miss                 Counter         request         # of LTS requests from unit L1 for reads marked evict-normal (LRU)    
                                                                                                            that missed                                                           
lts__t_requests_srcunit_l1_op_read_lookup_hit                               Counter         request         # of LTS requests from unit L1 for reads that hit                     
lts__t_requests_srcunit_l1_op_read_lookup_miss                              Counter         request         # of LTS requests from unit L1 for reads that missed                  
lts__t_requests_srcunit_l1_op_red                                           Counter         request         # of LTS requests from unit L1 for reductions                         
lts__t_requests_srcunit_l1_op_red_lookup_hit                                Counter         request         # of LTS requests from unit L1 for reductions that hit                
lts__t_requests_srcunit_l1_op_red_lookup_miss                               Counter         request         # of LTS requests from unit L1 for reductions that missed             
lts__t_requests_srcunit_l1_op_write                                         Counter         request         # of LTS requests from unit L1 for writes                             
lts__t_requests_srcunit_l1_op_write_evict_first                             Counter         request         # of LTS requests from unit L1 for writes marked evict-first          
lts__t_requests_srcunit_l1_op_write_evict_first_lookup_hit                  Counter         request         # of LTS requests from unit L1 for writes marked evict-first that hit 
lts__t_requests_srcunit_l1_op_write_evict_first_lookup_miss                 Counter         request         # of LTS requests from unit L1 for writes marked evict-first that     
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_op_write_evict_last                              Counter         request         # of LTS requests from unit L1 for writes marked evict-last           
lts__t_requests_srcunit_l1_op_write_evict_last_lookup_hit                   Counter         request         # of LTS requests from unit L1 for writes marked evict-last that hit  
lts__t_requests_srcunit_l1_op_write_evict_last_lookup_miss                  Counter         request         # of LTS requests from unit L1 for writes marked evict-last that      
                                                                                                            missed                                                                
lts__t_requests_srcunit_l1_op_write_evict_normal                            Counter         request         # of LTS requests from unit L1 for writes marked evict-normal (LRU)   
lts__t_requests_srcunit_l1_op_write_evict_normal_demote                     Counter         request         # of LTS requests from unit L1 for writes marked evict-normal-demote  
lts__t_requests_srcunit_l1_op_write_evict_normal_lookup_hit                 Counter         request         # of LTS requests from unit L1 for writes marked evict-normal (LRU)   
                                                                                                            that hit                                                              
lts__t_requests_srcunit_l1_op_write_evict_normal_lookup_miss                Counter         request         # of LTS requests from unit L1 for writes marked evict-normal (LRU)   
                                                                                                            that missed                                                           
lts__t_requests_srcunit_l1_op_write_lookup_hit                              Counter         request         # of LTS requests from unit L1 for writes that hit                    
lts__t_requests_srcunit_l1_op_write_lookup_miss                             Counter         request         # of LTS requests from unit L1 for writes that missed                 
lts__t_requests_srcunit_ltcfabric                                           Counter         request         # of LTS requests from LTC Fabric                                     
lts__t_requests_srcunit_ltcfabric_aperture_device                           Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
lts__t_requests_srcunit_ltcfabric_aperture_device_evict_first               Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_ltcfabric_aperture_device_evict_last                Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_ltcfabric_aperture_device_evict_normal              Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_ltcfabric_aperture_device_evict_normal_demote       Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_ltcfabric_aperture_device_op_membar                 Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            for memory barriers                                                   
lts__t_requests_srcunit_ltcfabric_aperture_device_op_read                   Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            for reads                                                             
lts__t_requests_srcunit_ltcfabric_aperture_device_op_write                  Counter         request         # of LTS requests from LTC Fabric accessing device memory (vidmem)    
                                                                                                            for writes                                                            
lts__t_requests_srcunit_ltcfabric_aperture_peer                             Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem)     
lts__t_requests_srcunit_ltcfabric_aperture_peer_evict_first                 Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem)     
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_ltcfabric_aperture_peer_evict_last                  Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem)     
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_ltcfabric_aperture_peer_evict_normal                Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem)     
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_ltcfabric_aperture_peer_evict_normal_demote         Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem)     
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_ltcfabric_aperture_peer_op_membar                   Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem) for 
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_ltcfabric_aperture_peer_op_read                     Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem) for 
                                                                                                            reads                                                                 
lts__t_requests_srcunit_ltcfabric_aperture_peer_op_write                    Counter         request         # of LTS requests from LTC Fabric accessing peer memory (peermem) for 
                                                                                                            writes                                                                
lts__t_requests_srcunit_ltcfabric_aperture_sysmem                           Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_evict_first               Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_evict_last                Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_evict_normal              Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_evict_normal_demote       Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_op_membar                 Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            for memory barriers                                                   
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_op_read                   Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            for reads                                                             
lts__t_requests_srcunit_ltcfabric_aperture_sysmem_op_write                  Counter         request         # of LTS requests from LTC Fabric accessing system memory (sysmem)    
                                                                                                            for writes                                                            
lts__t_requests_srcunit_ltcfabric_evict_first                               Counter         request         # of LTS requests from LTC Fabric marked evict-first                  
lts__t_requests_srcunit_ltcfabric_evict_last                                Counter         request         # of LTS requests from LTC Fabric marked evict-last                   
lts__t_requests_srcunit_ltcfabric_evict_normal                              Counter         request         # of LTS requests from LTC Fabric marked evict-normal (LRU)           
lts__t_requests_srcunit_ltcfabric_evict_normal_demote                       Counter         request         # of LTS requests from LTC Fabric marked evict-normal-demote          
lts__t_requests_srcunit_ltcfabric_op_membar                                 Counter         request         # of LTS requests from LTC Fabric for memory barriers                 
lts__t_requests_srcunit_ltcfabric_op_membar_evict_first                     Counter         request         # of LTS requests from LTC Fabric for memory barriers marked          
                                                                                                            evict-first                                                           
lts__t_requests_srcunit_ltcfabric_op_membar_evict_last                      Counter         request         # of LTS requests from LTC Fabric for memory barriers marked          
                                                                                                            evict-last                                                            
lts__t_requests_srcunit_ltcfabric_op_membar_evict_normal                    Counter         request         # of LTS requests from LTC Fabric for memory barriers marked          
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_srcunit_ltcfabric_op_membar_evict_normal_demote             Counter         request         # of LTS requests from LTC Fabric for memory barriers marked          
                                                                                                            evict-normal-demote                                                   
lts__t_requests_srcunit_ltcfabric_op_read                                   Counter         request         # of LTS requests from LTC Fabric for reads                           
lts__t_requests_srcunit_ltcfabric_op_read_evict_first                       Counter         request         # of LTS requests from LTC Fabric for reads marked evict-first        
lts__t_requests_srcunit_ltcfabric_op_read_evict_last                        Counter         request         # of LTS requests from LTC Fabric for reads marked evict-last         
lts__t_requests_srcunit_ltcfabric_op_read_evict_normal                      Counter         request         # of LTS requests from LTC Fabric for reads marked evict-normal (LRU) 
lts__t_requests_srcunit_ltcfabric_op_read_evict_normal_demote               Counter         request         # of LTS requests from LTC Fabric for reads marked evict-normal-demote
lts__t_requests_srcunit_ltcfabric_op_write                                  Counter         request         # of LTS requests from LTC Fabric for writes                          
lts__t_requests_srcunit_ltcfabric_op_write_evict_first                      Counter         request         # of LTS requests from LTC Fabric for writes marked evict-first       
lts__t_requests_srcunit_ltcfabric_op_write_evict_last                       Counter         request         # of LTS requests from LTC Fabric for writes marked evict-last        
lts__t_requests_srcunit_ltcfabric_op_write_evict_normal                     Counter         request         # of LTS requests from LTC Fabric for writes marked evict-normal (LRU)
lts__t_requests_srcunit_ltcfabric_op_write_evict_normal_demote              Counter         request         # of LTS requests from LTC Fabric for writes marked                   
                                                                                                            evict-normal-demote                                                   
lts__t_requests_srcunit_tex                                                 Counter         request         # of LTS requests from unit TEX                                       
lts__t_requests_srcunit_tex_aperture_device                                 Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
lts__t_requests_srcunit_tex_aperture_device_evict_first                     Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_tex_aperture_device_evict_first_lookup_hit          Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcunit_tex_aperture_device_evict_first_lookup_miss         Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcunit_tex_aperture_device_evict_last                      Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_tex_aperture_device_evict_last_lookup_hit           Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcunit_tex_aperture_device_evict_last_lookup_miss          Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcunit_tex_aperture_device_evict_normal                    Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_tex_aperture_device_evict_normal_demote             Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_tex_aperture_device_evict_normal_lookup_hit         Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcunit_tex_aperture_device_evict_normal_lookup_miss        Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem)      
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcunit_tex_aperture_device_lookup_hit                      Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) that 
                                                                                                            hit                                                                   
lts__t_requests_srcunit_tex_aperture_device_lookup_miss                     Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) that 
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_aperture_device_op_atom                         Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            all atomics                                                           
lts__t_requests_srcunit_tex_aperture_device_op_atom_dot_alu                 Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcunit_tex_aperture_device_op_atom_dot_alu_lookup_hit      Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcunit_tex_aperture_device_op_atom_dot_cas                 Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            atomic CAS                                                            
lts__t_requests_srcunit_tex_aperture_device_op_atom_dot_cas_lookup_hit      Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcunit_tex_aperture_device_op_atom_lookup_hit              Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcunit_tex_aperture_device_op_atom_lookup_miss             Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            all atomics that missed                                               
lts__t_requests_srcunit_tex_aperture_device_op_membar                       Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_tex_aperture_device_op_membar_lookup_hit            Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcunit_tex_aperture_device_op_membar_lookup_miss           Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcunit_tex_aperture_device_op_read                         Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            reads                                                                 
lts__t_requests_srcunit_tex_aperture_device_op_read_lookup_hit              Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            reads that hit                                                        
lts__t_requests_srcunit_tex_aperture_device_op_read_lookup_miss             Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            reads that missed                                                     
lts__t_requests_srcunit_tex_aperture_device_op_red                          Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            reductions                                                            
lts__t_requests_srcunit_tex_aperture_device_op_red_lookup_hit               Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            reductions that hit                                                   
lts__t_requests_srcunit_tex_aperture_device_op_red_lookup_miss              Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            reductions that missed                                                
lts__t_requests_srcunit_tex_aperture_device_op_write                        Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            writes                                                                
lts__t_requests_srcunit_tex_aperture_device_op_write_lookup_hit             Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            writes that hit                                                       
lts__t_requests_srcunit_tex_aperture_device_op_write_lookup_miss            Counter         request         # of LTS requests from unit TEX accessing device memory (vidmem) for  
                                                                                                            writes that missed                                                    
lts__t_requests_srcunit_tex_aperture_peer                                   Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
lts__t_requests_srcunit_tex_aperture_peer_evict_first                       Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_tex_aperture_peer_evict_first_lookup_hit            Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcunit_tex_aperture_peer_evict_first_lookup_miss           Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcunit_tex_aperture_peer_evict_last                        Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_tex_aperture_peer_evict_last_lookup_hit             Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcunit_tex_aperture_peer_evict_last_lookup_miss            Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcunit_tex_aperture_peer_evict_normal                      Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_tex_aperture_peer_evict_normal_demote               Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_tex_aperture_peer_evict_normal_lookup_hit           Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcunit_tex_aperture_peer_evict_normal_lookup_miss          Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcunit_tex_aperture_peer_lookup_hit                        Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) that  
                                                                                                            hit                                                                   
lts__t_requests_srcunit_tex_aperture_peer_lookup_miss                       Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) that  
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_aperture_peer_op_atom                           Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            all atomics                                                           
lts__t_requests_srcunit_tex_aperture_peer_op_atom_dot_alu                   Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcunit_tex_aperture_peer_op_atom_dot_alu_lookup_hit        Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcunit_tex_aperture_peer_op_atom_dot_cas                   Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            atomic CAS                                                            
lts__t_requests_srcunit_tex_aperture_peer_op_atom_dot_cas_lookup_hit        Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcunit_tex_aperture_peer_op_atom_lookup_hit                Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcunit_tex_aperture_peer_op_atom_lookup_miss               Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            all atomics that missed                                               
lts__t_requests_srcunit_tex_aperture_peer_op_membar                         Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_tex_aperture_peer_op_membar_lookup_hit              Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcunit_tex_aperture_peer_op_membar_lookup_miss             Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcunit_tex_aperture_peer_op_read                           Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            reads                                                                 
lts__t_requests_srcunit_tex_aperture_peer_op_read_lookup_hit                Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            reads that hit                                                        
lts__t_requests_srcunit_tex_aperture_peer_op_read_lookup_miss               Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            reads that missed                                                     
lts__t_requests_srcunit_tex_aperture_peer_op_red                            Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            reductions                                                            
lts__t_requests_srcunit_tex_aperture_peer_op_red_lookup_hit                 Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            reductions that hit                                                   
lts__t_requests_srcunit_tex_aperture_peer_op_red_lookup_miss                Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            reductions that missed                                                
lts__t_requests_srcunit_tex_aperture_peer_op_write                          Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            writes                                                                
lts__t_requests_srcunit_tex_aperture_peer_op_write_lookup_hit               Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            writes that hit                                                       
lts__t_requests_srcunit_tex_aperture_peer_op_write_lookup_miss              Counter         request         # of LTS requests from unit TEX accessing peer memory (peermem) for   
                                                                                                            writes that missed                                                    
lts__t_requests_srcunit_tex_aperture_sysmem                                 Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
lts__t_requests_srcunit_tex_aperture_sysmem_evict_first                     Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-first                                                    
lts__t_requests_srcunit_tex_aperture_sysmem_evict_first_lookup_hit          Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-first that hit                                           
lts__t_requests_srcunit_tex_aperture_sysmem_evict_first_lookup_miss         Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-first that missed                                        
lts__t_requests_srcunit_tex_aperture_sysmem_evict_last                      Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-last                                                     
lts__t_requests_srcunit_tex_aperture_sysmem_evict_last_lookup_hit           Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-last that hit                                            
lts__t_requests_srcunit_tex_aperture_sysmem_evict_last_lookup_miss          Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-last that missed                                         
lts__t_requests_srcunit_tex_aperture_sysmem_evict_normal                    Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-normal (LRU)                                             
lts__t_requests_srcunit_tex_aperture_sysmem_evict_normal_demote             Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-normal-demote                                            
lts__t_requests_srcunit_tex_aperture_sysmem_evict_normal_lookup_hit         Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_requests_srcunit_tex_aperture_sysmem_evict_normal_lookup_miss        Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem)      
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_requests_srcunit_tex_aperture_sysmem_lookup_hit                      Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) that 
                                                                                                            hit                                                                   
lts__t_requests_srcunit_tex_aperture_sysmem_lookup_miss                     Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) that 
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom                         Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            all atomics                                                           
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom_dot_alu                 Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom_dot_alu_lookup_hit      Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom_dot_cas                 Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            atomic CAS                                                            
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom_dot_cas_lookup_hit      Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            atomic CAS that hit                                                   
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom_lookup_hit              Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            all atomics that hit                                                  
lts__t_requests_srcunit_tex_aperture_sysmem_op_atom_lookup_miss             Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            all atomics that missed                                               
lts__t_requests_srcunit_tex_aperture_sysmem_op_membar                       Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            memory barriers                                                       
lts__t_requests_srcunit_tex_aperture_sysmem_op_membar_lookup_hit            Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            memory barriers that hit                                              
lts__t_requests_srcunit_tex_aperture_sysmem_op_membar_lookup_miss           Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            memory barriers that missed                                           
lts__t_requests_srcunit_tex_aperture_sysmem_op_read                         Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            reads                                                                 
lts__t_requests_srcunit_tex_aperture_sysmem_op_read_lookup_hit              Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            reads that hit                                                        
lts__t_requests_srcunit_tex_aperture_sysmem_op_read_lookup_miss             Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            reads that missed                                                     
lts__t_requests_srcunit_tex_aperture_sysmem_op_red                          Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            reductions                                                            
lts__t_requests_srcunit_tex_aperture_sysmem_op_red_lookup_hit               Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            reductions that hit                                                   
lts__t_requests_srcunit_tex_aperture_sysmem_op_red_lookup_miss              Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            reductions that missed                                                
lts__t_requests_srcunit_tex_aperture_sysmem_op_write                        Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            writes                                                                
lts__t_requests_srcunit_tex_aperture_sysmem_op_write_lookup_hit             Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            writes that hit                                                       
lts__t_requests_srcunit_tex_aperture_sysmem_op_write_lookup_miss            Counter         request         # of LTS requests from unit TEX accessing system memory (sysmem) for  
                                                                                                            writes that missed                                                    
lts__t_requests_srcunit_tex_evict_first                                     Counter         request         # of LTS requests from unit TEX marked evict-first                    
lts__t_requests_srcunit_tex_evict_first_lookup_hit                          Counter         request         # of LTS requests from unit TEX marked evict-first that hit           
lts__t_requests_srcunit_tex_evict_first_lookup_miss                         Counter         request         # of LTS requests from unit TEX marked evict-first that missed        
lts__t_requests_srcunit_tex_evict_last                                      Counter         request         # of LTS requests from unit TEX marked evict-last                     
lts__t_requests_srcunit_tex_evict_last_lookup_hit                           Counter         request         # of LTS requests from unit TEX marked evict-last that hit            
lts__t_requests_srcunit_tex_evict_last_lookup_miss                          Counter         request         # of LTS requests from unit TEX marked evict-last that missed         
lts__t_requests_srcunit_tex_evict_normal                                    Counter         request         # of LTS requests from unit TEX marked evict-normal (LRU)             
lts__t_requests_srcunit_tex_evict_normal_demote                             Counter         request         # of LTS requests from unit TEX marked evict-normal-demote            
lts__t_requests_srcunit_tex_evict_normal_demote_lookup_hit                  Counter         request         # of LTS requests from unit TEX marked evict-normal-demote that hit   
lts__t_requests_srcunit_tex_evict_normal_demote_lookup_miss                 Counter         request         # of LTS requests from unit TEX marked evict-normal-demote that missed
lts__t_requests_srcunit_tex_evict_normal_lookup_hit                         Counter         request         # of LTS requests from unit TEX marked evict-normal (LRU) that hit    
lts__t_requests_srcunit_tex_evict_normal_lookup_miss                        Counter         request         # of LTS requests from unit TEX marked evict-normal (LRU) that missed 
lts__t_requests_srcunit_tex_lookup_hit                                      Counter         request         # of LTS requests from unit TEX that hit                              
lts__t_requests_srcunit_tex_lookup_miss                                     Counter         request         # of LTS requests from unit TEX that missed                           
lts__t_requests_srcunit_tex_op_atom                                         Counter         request         # of LTS requests from unit TEX for all atomics                       
lts__t_requests_srcunit_tex_op_atom_dot_alu                                 Counter         request         # of LTS requests from unit TEX for atomic ALU (non-CAS)              
lts__t_requests_srcunit_tex_op_atom_dot_alu_lookup_hit                      Counter         request         # of LTS requests from unit TEX for atomic ALU (non-CAS) that hit     
lts__t_requests_srcunit_tex_op_atom_dot_cas                                 Counter         request         # of LTS requests from unit TEX for atomic CAS                        
lts__t_requests_srcunit_tex_op_atom_dot_cas_lookup_hit                      Counter         request         # of LTS requests from unit TEX for atomic CAS that hit               
lts__t_requests_srcunit_tex_op_atom_evict_first                             Counter         request         # of LTS requests from unit TEX for all atomics marked evict-first    
lts__t_requests_srcunit_tex_op_atom_evict_first_lookup_hit                  Counter         request         # of LTS requests from unit TEX for all atomics marked evict-first    
                                                                                                            that hit                                                              
lts__t_requests_srcunit_tex_op_atom_evict_first_lookup_miss                 Counter         request         # of LTS requests from unit TEX for all atomics marked evict-first    
                                                                                                            that missed                                                           
lts__t_requests_srcunit_tex_op_atom_evict_last                              Counter         request         # of LTS requests from unit TEX for all atomics marked evict-last     
lts__t_requests_srcunit_tex_op_atom_evict_last_lookup_hit                   Counter         request         # of LTS requests from unit TEX for all atomics marked evict-last     
                                                                                                            that hit                                                              
lts__t_requests_srcunit_tex_op_atom_evict_last_lookup_miss                  Counter         request         # of LTS requests from unit TEX for all atomics marked evict-last     
                                                                                                            that missed                                                           
lts__t_requests_srcunit_tex_op_atom_evict_normal                            Counter         request         # of LTS requests from unit TEX for all atomics marked evict-normal   
                                                                                                            (LRU)                                                                 
lts__t_requests_srcunit_tex_op_atom_evict_normal_lookup_hit                 Counter         request         # of LTS requests from unit TEX for all atomics marked evict-normal   
                                                                                                            (LRU) that hit                                                        
lts__t_requests_srcunit_tex_op_atom_evict_normal_lookup_miss                Counter         request         # of LTS requests from unit TEX for all atomics marked evict-normal   
                                                                                                            (LRU) that missed                                                     
lts__t_requests_srcunit_tex_op_atom_lookup_hit                              Counter         request         # of LTS requests from unit TEX for all atomics that hit              
lts__t_requests_srcunit_tex_op_atom_lookup_miss                             Counter         request         # of LTS requests from unit TEX for all atomics that missed           
lts__t_requests_srcunit_tex_op_membar                                       Counter         request         # of LTS requests from unit TEX for memory barriers                   
lts__t_requests_srcunit_tex_op_membar_evict_first                           Counter         request         # of LTS requests from unit TEX for memory barriers marked evict-first
lts__t_requests_srcunit_tex_op_membar_evict_first_lookup_hit                Counter         request         # of LTS requests from unit TEX for memory barriers marked            
                                                                                                            evict-first that hit                                                  
lts__t_requests_srcunit_tex_op_membar_evict_first_lookup_miss               Counter         request         # of LTS requests from unit TEX for memory barriers marked            
                                                                                                            evict-first that missed                                               
lts__t_requests_srcunit_tex_op_membar_evict_last                            Counter         request         # of LTS requests from unit TEX for memory barriers marked evict-last 
lts__t_requests_srcunit_tex_op_membar_evict_last_lookup_hit                 Counter         request         # of LTS requests from unit TEX for memory barriers marked evict-last 
                                                                                                            that hit                                                              
lts__t_requests_srcunit_tex_op_membar_evict_last_lookup_miss                Counter         request         # of LTS requests from unit TEX for memory barriers marked evict-last 
                                                                                                            that missed                                                           
lts__t_requests_srcunit_tex_op_membar_evict_normal                          Counter         request         # of LTS requests from unit TEX for memory barriers marked            
                                                                                                            evict-normal (LRU)                                                    
lts__t_requests_srcunit_tex_op_membar_evict_normal_demote                   Counter         request         # of LTS requests from unit TEX for memory barriers marked            
                                                                                                            evict-normal-demote                                                   
lts__t_requests_srcunit_tex_op_membar_evict_normal_lookup_hit               Counter         request         # of LTS requests from unit TEX for memory barriers marked            
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_requests_srcunit_tex_op_membar_evict_normal_lookup_miss              Counter         request         # of LTS requests from unit TEX for memory barriers marked            
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_requests_srcunit_tex_op_membar_lookup_hit                            Counter         request         # of LTS requests from unit TEX for memory barriers that hit          
lts__t_requests_srcunit_tex_op_membar_lookup_miss                           Counter         request         # of LTS requests from unit TEX for memory barriers that missed       
lts__t_requests_srcunit_tex_op_read                                         Counter         request         # of LTS requests from unit TEX for reads                             
lts__t_requests_srcunit_tex_op_read_evict_first                             Counter         request         # of LTS requests from unit TEX for reads marked evict-first          
lts__t_requests_srcunit_tex_op_read_evict_first_lookup_hit                  Counter         request         # of LTS requests from unit TEX for reads marked evict-first that hit 
lts__t_requests_srcunit_tex_op_read_evict_first_lookup_miss                 Counter         request         # of LTS requests from unit TEX for reads marked evict-first that     
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_op_read_evict_last                              Counter         request         # of LTS requests from unit TEX for reads marked evict-last           
lts__t_requests_srcunit_tex_op_read_evict_last_lookup_hit                   Counter         request         # of LTS requests from unit TEX for reads marked evict-last that hit  
lts__t_requests_srcunit_tex_op_read_evict_last_lookup_miss                  Counter         request         # of LTS requests from unit TEX for reads marked evict-last that      
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_op_read_evict_normal                            Counter         request         # of LTS requests from unit TEX for reads marked evict-normal (LRU)   
lts__t_requests_srcunit_tex_op_read_evict_normal_demote                     Counter         request         # of LTS requests from unit TEX for reads marked evict-normal-demote  
lts__t_requests_srcunit_tex_op_read_evict_normal_lookup_hit                 Counter         request         # of LTS requests from unit TEX for reads marked evict-normal (LRU)   
                                                                                                            that hit                                                              
lts__t_requests_srcunit_tex_op_read_evict_normal_lookup_miss                Counter         request         # of LTS requests from unit TEX for reads marked evict-normal (LRU)   
                                                                                                            that missed                                                           
lts__t_requests_srcunit_tex_op_read_lookup_hit                              Counter         request         # of LTS requests from unit TEX for reads that hit                    
lts__t_requests_srcunit_tex_op_read_lookup_miss                             Counter         request         # of LTS requests from unit TEX for reads that missed                 
lts__t_requests_srcunit_tex_op_red                                          Counter         request         # of LTS requests from unit TEX for reductions                        
lts__t_requests_srcunit_tex_op_red_lookup_hit                               Counter         request         # of LTS requests from unit TEX for reductions that hit               
lts__t_requests_srcunit_tex_op_red_lookup_miss                              Counter         request         # of LTS requests from unit TEX for reductions that missed            
lts__t_requests_srcunit_tex_op_write                                        Counter         request         # of LTS requests from unit TEX for writes                            
lts__t_requests_srcunit_tex_op_write_evict_first                            Counter         request         # of LTS requests from unit TEX for writes marked evict-first         
lts__t_requests_srcunit_tex_op_write_evict_first_lookup_hit                 Counter         request         # of LTS requests from unit TEX for writes marked evict-first that hit
lts__t_requests_srcunit_tex_op_write_evict_first_lookup_miss                Counter         request         # of LTS requests from unit TEX for writes marked evict-first that    
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_op_write_evict_last                             Counter         request         # of LTS requests from unit TEX for writes marked evict-last          
lts__t_requests_srcunit_tex_op_write_evict_last_lookup_hit                  Counter         request         # of LTS requests from unit TEX for writes marked evict-last that hit 
lts__t_requests_srcunit_tex_op_write_evict_last_lookup_miss                 Counter         request         # of LTS requests from unit TEX for writes marked evict-last that     
                                                                                                            missed                                                                
lts__t_requests_srcunit_tex_op_write_evict_normal                           Counter         request         # of LTS requests from unit TEX for writes marked evict-normal (LRU)  
lts__t_requests_srcunit_tex_op_write_evict_normal_demote                    Counter         request         # of LTS requests from unit TEX for writes marked evict-normal-demote 
lts__t_requests_srcunit_tex_op_write_evict_normal_lookup_hit                Counter         request         # of LTS requests from unit TEX for writes marked evict-normal (LRU)  
                                                                                                            that hit                                                              
lts__t_requests_srcunit_tex_op_write_evict_normal_lookup_miss               Counter         request         # of LTS requests from unit TEX for writes marked evict-normal (LRU)  
                                                                                                            that missed                                                           
lts__t_requests_srcunit_tex_op_write_lookup_hit                             Counter         request         # of LTS requests from unit TEX for writes that hit                   
lts__t_requests_srcunit_tex_op_write_lookup_miss                            Counter         request         # of LTS requests from unit TEX for writes that missed                
lts__t_sector_hit_rate                                                      Ratio                           proportion of L2 sector lookups that hit                              
lts__t_sector_op_atom_dot_alu_hit_rate                                      Ratio                           # of sector hits for _op_atom_dot_alu per sector access for           
                                                                                                            _op_atom_dot_alu                                                      
lts__t_sector_op_atom_dot_cas_hit_rate                                      Ratio                           # of sector hits for _op_atom_dot_cas per sector access for           
                                                                                                            _op_atom_dot_cas                                                      
lts__t_sector_op_atom_hit_rate                                              Ratio                           # of sector hits for _op_atom per sector access for _op_atom          
lts__t_sector_op_read_hit_rate                                              Ratio                           # of sector hits for _op_read per sector access for _op_read          
lts__t_sector_op_red_hit_rate                                               Ratio                           # of sector hits for _op_red per sector access for _op_red            
lts__t_sector_op_write_hit_rate                                             Ratio                           # of sector hits for _op_write per sector access for _op_write        
lts__t_sectors                                                              Counter         sector          # of LTS sectors                                                      
lts__t_sectors_aperture_device                                              Counter         sector          # of LTS sectors accessing device memory (vidmem)                     
lts__t_sectors_aperture_device_evict_first                                  Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-first  
lts__t_sectors_aperture_device_evict_first_lookup_hit                       Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-first  
                                                                                                            that hit                                                              
lts__t_sectors_aperture_device_evict_first_lookup_miss                      Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-first  
                                                                                                            that missed                                                           
lts__t_sectors_aperture_device_evict_last                                   Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-last   
lts__t_sectors_aperture_device_evict_last_lookup_hit                        Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-last   
                                                                                                            that hit                                                              
lts__t_sectors_aperture_device_evict_last_lookup_miss                       Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-last   
                                                                                                            that missed                                                           
lts__t_sectors_aperture_device_evict_normal                                 Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-normal 
                                                                                                            (LRU)                                                                 
lts__t_sectors_aperture_device_evict_normal_demote                          Counter         sector          # of LTS sectors accessing device memory (vidmem) marked              
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_device_evict_normal_demote_lookup_hit               Counter         sector          # of LTS sectors accessing device memory (vidmem) marked              
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_device_evict_normal_demote_lookup_miss              Counter         sector          # of LTS sectors accessing device memory (vidmem) marked              
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_device_evict_normal_lookup_hit                      Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-normal 
                                                                                                            (LRU) that hit                                                        
lts__t_sectors_aperture_device_evict_normal_lookup_miss                     Counter         sector          # of LTS sectors accessing device memory (vidmem) marked evict-normal 
                                                                                                            (LRU) that missed                                                     
lts__t_sectors_aperture_device_lookup_hit                                   Counter         sector          # of LTS sectors accessing device memory (vidmem) that hit            
lts__t_sectors_aperture_device_lookup_miss                                  Counter         sector          # of LTS sectors accessing device memory (vidmem) that missed         
lts__t_sectors_aperture_device_op_atom                                      Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
lts__t_sectors_aperture_device_op_atom_dot_alu                              Counter         sector          # of LTS sectors accessing device memory (vidmem) for atomic ALU      
                                                                                                            (non-CAS)                                                             
lts__t_sectors_aperture_device_op_atom_dot_alu_lookup_hit                   Counter         sector          # of LTS sectors accessing device memory (vidmem) for atomic ALU      
                                                                                                            (non-CAS) that hit                                                    
lts__t_sectors_aperture_device_op_atom_dot_alu_lookup_miss                  Counter         sector          # of LTS sectors accessing device memory (vidmem) for atomic ALU      
                                                                                                            (non-CAS) that missed                                                 
lts__t_sectors_aperture_device_op_atom_dot_cas                              Counter         sector          # of LTS sectors accessing device memory (vidmem) for atomic CAS      
lts__t_sectors_aperture_device_op_atom_dot_cas_lookup_hit                   Counter         sector          # of LTS sectors accessing device memory (vidmem) for atomic CAS that 
                                                                                                            hit                                                                   
lts__t_sectors_aperture_device_op_atom_dot_cas_lookup_miss                  Counter         sector          # of LTS sectors accessing device memory (vidmem) for atomic CAS that 
                                                                                                            missed                                                                
lts__t_sectors_aperture_device_op_atom_evict_first                          Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-first                                                    
lts__t_sectors_aperture_device_op_atom_evict_first_lookup_hit               Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_aperture_device_op_atom_evict_first_lookup_miss              Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_aperture_device_op_atom_evict_last                           Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-last                                                     
lts__t_sectors_aperture_device_op_atom_evict_last_lookup_hit                Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_aperture_device_op_atom_evict_last_lookup_miss               Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_aperture_device_op_atom_evict_normal                         Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_aperture_device_op_atom_evict_normal_lookup_hit              Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_aperture_device_op_atom_evict_normal_lookup_miss             Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_aperture_device_op_atom_lookup_hit                           Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            that hit                                                              
lts__t_sectors_aperture_device_op_atom_lookup_miss                          Counter         sector          # of LTS sectors accessing device memory (vidmem) for all atomics     
                                                                                                            that missed                                                           
lts__t_sectors_aperture_device_op_membar                                    Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
lts__t_sectors_aperture_device_op_membar_evict_first                        Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-first                                                    
lts__t_sectors_aperture_device_op_membar_evict_first_lookup_hit             Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_aperture_device_op_membar_evict_first_lookup_miss            Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_aperture_device_op_membar_evict_last                         Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-last                                                     
lts__t_sectors_aperture_device_op_membar_evict_last_lookup_hit              Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_aperture_device_op_membar_evict_last_lookup_miss             Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_aperture_device_op_membar_evict_normal                       Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_aperture_device_op_membar_evict_normal_demote                Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_aperture_device_op_membar_evict_normal_demote_lookup_hit     Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_aperture_device_op_membar_evict_normal_demote_lookup_miss    Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_aperture_device_op_membar_evict_normal_lookup_hit            Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_aperture_device_op_membar_evict_normal_lookup_miss           Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_aperture_device_op_membar_lookup_hit                         Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            that hit                                                              
lts__t_sectors_aperture_device_op_membar_lookup_miss                        Counter         sector          # of LTS sectors accessing device memory (vidmem) for memory barriers 
                                                                                                            that missed                                                           
lts__t_sectors_aperture_device_op_read                                      Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads           
lts__t_sectors_aperture_device_op_read_evict_first                          Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-first                                                           
lts__t_sectors_aperture_device_op_read_evict_first_lookup_hit               Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-first that hit                                                  
lts__t_sectors_aperture_device_op_read_evict_first_lookup_miss              Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-first that missed                                               
lts__t_sectors_aperture_device_op_read_evict_last                           Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-last                                                            
lts__t_sectors_aperture_device_op_read_evict_last_lookup_hit                Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-last that hit                                                   
lts__t_sectors_aperture_device_op_read_evict_last_lookup_miss               Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-last that missed                                                
lts__t_sectors_aperture_device_op_read_evict_normal                         Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_aperture_device_op_read_evict_normal_demote                  Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_device_op_read_evict_normal_demote_lookup_hit       Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_device_op_read_evict_normal_demote_lookup_miss      Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_device_op_read_evict_normal_lookup_hit              Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_aperture_device_op_read_evict_normal_lookup_miss             Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads marked    
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_aperture_device_op_read_lookup_hit                           Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads that hit  
lts__t_sectors_aperture_device_op_read_lookup_miss                          Counter         sector          # of LTS sectors accessing device memory (vidmem) for reads that      
                                                                                                            missed                                                                
lts__t_sectors_aperture_device_op_red                                       Counter         sector          # of LTS sectors accessing device memory (vidmem) for reductions      
lts__t_sectors_aperture_device_op_red_lookup_hit                            Counter         sector          # of LTS sectors accessing device memory (vidmem) for reductions that 
                                                                                                            hit                                                                   
lts__t_sectors_aperture_device_op_red_lookup_miss                           Counter         sector          # of LTS sectors accessing device memory (vidmem) for reductions that 
                                                                                                            missed                                                                
lts__t_sectors_aperture_device_op_write                                     Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes          
lts__t_sectors_aperture_device_op_write_evict_first                         Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-first                                                           
lts__t_sectors_aperture_device_op_write_evict_first_lookup_hit              Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-first that hit                                                  
lts__t_sectors_aperture_device_op_write_evict_first_lookup_miss             Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-first that missed                                               
lts__t_sectors_aperture_device_op_write_evict_last                          Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-last                                                            
lts__t_sectors_aperture_device_op_write_evict_last_lookup_hit               Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-last that hit                                                   
lts__t_sectors_aperture_device_op_write_evict_last_lookup_miss              Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-last that missed                                                
lts__t_sectors_aperture_device_op_write_evict_normal                        Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_aperture_device_op_write_evict_normal_demote                 Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_device_op_write_evict_normal_demote_lookup_hit      Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_device_op_write_evict_normal_demote_lookup_miss     Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_device_op_write_evict_normal_lookup_hit             Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_aperture_device_op_write_evict_normal_lookup_miss            Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes marked   
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_aperture_device_op_write_lookup_hit                          Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes that hit 
lts__t_sectors_aperture_device_op_write_lookup_miss                         Counter         sector          # of LTS sectors accessing device memory (vidmem) for writes that     
                                                                                                            missed                                                                
lts__t_sectors_aperture_peer                                                Counter         sector          # of LTS sectors accessing peer memory (peermem)                      
lts__t_sectors_aperture_peer_evict_first                                    Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-first   
lts__t_sectors_aperture_peer_evict_first_lookup_hit                         Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-first   
                                                                                                            that hit                                                              
lts__t_sectors_aperture_peer_evict_first_lookup_miss                        Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-first   
                                                                                                            that missed                                                           
lts__t_sectors_aperture_peer_evict_last                                     Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-last    
lts__t_sectors_aperture_peer_evict_last_lookup_hit                          Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-last    
                                                                                                            that hit                                                              
lts__t_sectors_aperture_peer_evict_last_lookup_miss                         Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-last    
                                                                                                            that missed                                                           
lts__t_sectors_aperture_peer_evict_normal                                   Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-normal  
                                                                                                            (LRU)                                                                 
lts__t_sectors_aperture_peer_evict_normal_demote                            Counter         sector          # of LTS sectors accessing peer memory (peermem) marked               
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_peer_evict_normal_demote_lookup_hit                 Counter         sector          # of LTS sectors accessing peer memory (peermem) marked               
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_peer_evict_normal_demote_lookup_miss                Counter         sector          # of LTS sectors accessing peer memory (peermem) marked               
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_peer_evict_normal_lookup_hit                        Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-normal  
                                                                                                            (LRU) that hit                                                        
lts__t_sectors_aperture_peer_evict_normal_lookup_miss                       Counter         sector          # of LTS sectors accessing peer memory (peermem) marked evict-normal  
                                                                                                            (LRU) that missed                                                     
lts__t_sectors_aperture_peer_lookup_hit                                     Counter         sector          # of LTS sectors accessing peer memory (peermem) that hit             
lts__t_sectors_aperture_peer_lookup_miss                                    Counter         sector          # of LTS sectors accessing peer memory (peermem) that missed          
lts__t_sectors_aperture_peer_op_atom                                        Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
lts__t_sectors_aperture_peer_op_atom_dot_alu                                Counter         sector          # of LTS sectors accessing peer memory (peermem) for atomic ALU       
                                                                                                            (non-CAS)                                                             
lts__t_sectors_aperture_peer_op_atom_dot_alu_lookup_hit                     Counter         sector          # of LTS sectors accessing peer memory (peermem) for atomic ALU       
                                                                                                            (non-CAS) that hit                                                    
lts__t_sectors_aperture_peer_op_atom_dot_alu_lookup_miss                    Counter         sector          # of LTS sectors accessing peer memory (peermem) for atomic ALU       
                                                                                                            (non-CAS) that missed                                                 
lts__t_sectors_aperture_peer_op_atom_dot_cas                                Counter         sector          # of LTS sectors accessing peer memory (peermem) for atomic CAS       
lts__t_sectors_aperture_peer_op_atom_dot_cas_lookup_hit                     Counter         sector          # of LTS sectors accessing peer memory (peermem) for atomic CAS that  
                                                                                                            hit                                                                   
lts__t_sectors_aperture_peer_op_atom_dot_cas_lookup_miss                    Counter         sector          # of LTS sectors accessing peer memory (peermem) for atomic CAS that  
                                                                                                            missed                                                                
lts__t_sectors_aperture_peer_op_atom_evict_first                            Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-first                                                    
lts__t_sectors_aperture_peer_op_atom_evict_first_lookup_hit                 Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_aperture_peer_op_atom_evict_first_lookup_miss                Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_aperture_peer_op_atom_evict_last                             Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-last                                                     
lts__t_sectors_aperture_peer_op_atom_evict_last_lookup_hit                  Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_aperture_peer_op_atom_evict_last_lookup_miss                 Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_aperture_peer_op_atom_evict_normal                           Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_aperture_peer_op_atom_evict_normal_lookup_hit                Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_aperture_peer_op_atom_evict_normal_lookup_miss               Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics      
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_aperture_peer_op_atom_lookup_hit                             Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics that 
                                                                                                            hit                                                                   
lts__t_sectors_aperture_peer_op_atom_lookup_miss                            Counter         sector          # of LTS sectors accessing peer memory (peermem) for all atomics that 
                                                                                                            missed                                                                
lts__t_sectors_aperture_peer_op_membar                                      Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
lts__t_sectors_aperture_peer_op_membar_evict_first                          Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-first                                                    
lts__t_sectors_aperture_peer_op_membar_evict_first_lookup_hit               Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_aperture_peer_op_membar_evict_first_lookup_miss              Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_aperture_peer_op_membar_evict_last                           Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-last                                                     
lts__t_sectors_aperture_peer_op_membar_evict_last_lookup_hit                Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_aperture_peer_op_membar_evict_last_lookup_miss               Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_aperture_peer_op_membar_evict_normal                         Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_aperture_peer_op_membar_evict_normal_demote                  Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_aperture_peer_op_membar_evict_normal_demote_lookup_hit       Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_aperture_peer_op_membar_evict_normal_demote_lookup_miss      Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_aperture_peer_op_membar_evict_normal_lookup_hit              Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_aperture_peer_op_membar_evict_normal_lookup_miss             Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_aperture_peer_op_membar_lookup_hit                           Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            that hit                                                              
lts__t_sectors_aperture_peer_op_membar_lookup_miss                          Counter         sector          # of LTS sectors accessing peer memory (peermem) for memory barriers  
                                                                                                            that missed                                                           
lts__t_sectors_aperture_peer_op_read                                        Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads            
lts__t_sectors_aperture_peer_op_read_evict_first                            Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-first                                                           
lts__t_sectors_aperture_peer_op_read_evict_first_lookup_hit                 Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-first that hit                                                  
lts__t_sectors_aperture_peer_op_read_evict_first_lookup_miss                Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-first that missed                                               
lts__t_sectors_aperture_peer_op_read_evict_last                             Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-last                                                            
lts__t_sectors_aperture_peer_op_read_evict_last_lookup_hit                  Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-last that hit                                                   
lts__t_sectors_aperture_peer_op_read_evict_last_lookup_miss                 Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-last that missed                                                
lts__t_sectors_aperture_peer_op_read_evict_normal                           Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_aperture_peer_op_read_evict_normal_demote                    Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_peer_op_read_evict_normal_demote_lookup_hit         Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_peer_op_read_evict_normal_demote_lookup_miss        Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_peer_op_read_evict_normal_lookup_hit                Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_aperture_peer_op_read_evict_normal_lookup_miss               Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads marked     
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_aperture_peer_op_read_lookup_hit                             Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads that hit   
lts__t_sectors_aperture_peer_op_read_lookup_miss                            Counter         sector          # of LTS sectors accessing peer memory (peermem) for reads that missed
lts__t_sectors_aperture_peer_op_red                                         Counter         sector          # of LTS sectors accessing peer memory (peermem) for reductions       
lts__t_sectors_aperture_peer_op_red_lookup_hit                              Counter         sector          # of LTS sectors accessing peer memory (peermem) for reductions that  
                                                                                                            hit                                                                   
lts__t_sectors_aperture_peer_op_red_lookup_miss                             Counter         sector          # of LTS sectors accessing peer memory (peermem) for reductions that  
                                                                                                            missed                                                                
lts__t_sectors_aperture_peer_op_write                                       Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes           
lts__t_sectors_aperture_peer_op_write_evict_first                           Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-first                                                           
lts__t_sectors_aperture_peer_op_write_evict_first_lookup_hit                Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-first that hit                                                  
lts__t_sectors_aperture_peer_op_write_evict_first_lookup_miss               Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-first that missed                                               
lts__t_sectors_aperture_peer_op_write_evict_last                            Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-last                                                            
lts__t_sectors_aperture_peer_op_write_evict_last_lookup_hit                 Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-last that hit                                                   
lts__t_sectors_aperture_peer_op_write_evict_last_lookup_miss                Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-last that missed                                                
lts__t_sectors_aperture_peer_op_write_evict_normal                          Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_aperture_peer_op_write_evict_normal_demote                   Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_peer_op_write_evict_normal_demote_lookup_hit        Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_peer_op_write_evict_normal_demote_lookup_miss       Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_peer_op_write_evict_normal_lookup_hit               Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_aperture_peer_op_write_evict_normal_lookup_miss              Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes marked    
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_aperture_peer_op_write_lookup_hit                            Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes that hit  
lts__t_sectors_aperture_peer_op_write_lookup_miss                           Counter         sector          # of LTS sectors accessing peer memory (peermem) for writes that      
                                                                                                            missed                                                                
lts__t_sectors_aperture_sysmem                                              Counter         sector          # of LTS sectors accessing system memory (sysmem)                     
lts__t_sectors_aperture_sysmem_evict_first                                  Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-first  
lts__t_sectors_aperture_sysmem_evict_first_lookup_hit                       Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-first  
                                                                                                            that hit                                                              
lts__t_sectors_aperture_sysmem_evict_first_lookup_miss                      Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-first  
                                                                                                            that missed                                                           
lts__t_sectors_aperture_sysmem_evict_last                                   Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-last   
lts__t_sectors_aperture_sysmem_evict_last_lookup_hit                        Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-last   
                                                                                                            that hit                                                              
lts__t_sectors_aperture_sysmem_evict_last_lookup_miss                       Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-last   
                                                                                                            that missed                                                           
lts__t_sectors_aperture_sysmem_evict_normal                                 Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-normal 
                                                                                                            (LRU)                                                                 
lts__t_sectors_aperture_sysmem_evict_normal_demote                          Counter         sector          # of LTS sectors accessing system memory (sysmem) marked              
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_sysmem_evict_normal_demote_lookup_hit               Counter         sector          # of LTS sectors accessing system memory (sysmem) marked              
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_sysmem_evict_normal_demote_lookup_miss              Counter         sector          # of LTS sectors accessing system memory (sysmem) marked              
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_sysmem_evict_normal_lookup_hit                      Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-normal 
                                                                                                            (LRU) that hit                                                        
lts__t_sectors_aperture_sysmem_evict_normal_lookup_miss                     Counter         sector          # of LTS sectors accessing system memory (sysmem) marked evict-normal 
                                                                                                            (LRU) that missed                                                     
lts__t_sectors_aperture_sysmem_lookup_hit                                   Counter         sector          # of LTS sectors accessing system memory (sysmem) that hit            
lts__t_sectors_aperture_sysmem_lookup_miss                                  Counter         sector          # of LTS sectors accessing system memory (sysmem) that missed         
lts__t_sectors_aperture_sysmem_op_atom                                      Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
lts__t_sectors_aperture_sysmem_op_atom_dot_alu                              Counter         sector          # of LTS sectors accessing system memory (sysmem) for atomic ALU      
                                                                                                            (non-CAS)                                                             
lts__t_sectors_aperture_sysmem_op_atom_dot_alu_lookup_hit                   Counter         sector          # of LTS sectors accessing system memory (sysmem) for atomic ALU      
                                                                                                            (non-CAS) that hit                                                    
lts__t_sectors_aperture_sysmem_op_atom_dot_alu_lookup_miss                  Counter         sector          # of LTS sectors accessing system memory (sysmem) for atomic ALU      
                                                                                                            (non-CAS) that missed                                                 
lts__t_sectors_aperture_sysmem_op_atom_dot_cas                              Counter         sector          # of LTS sectors accessing system memory (sysmem) for atomic CAS      
lts__t_sectors_aperture_sysmem_op_atom_dot_cas_lookup_hit                   Counter         sector          # of LTS sectors accessing system memory (sysmem) for atomic CAS that 
                                                                                                            hit                                                                   
lts__t_sectors_aperture_sysmem_op_atom_dot_cas_lookup_miss                  Counter         sector          # of LTS sectors accessing system memory (sysmem) for atomic CAS that 
                                                                                                            missed                                                                
lts__t_sectors_aperture_sysmem_op_atom_evict_first                          Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-first                                                    
lts__t_sectors_aperture_sysmem_op_atom_evict_first_lookup_hit               Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_aperture_sysmem_op_atom_evict_first_lookup_miss              Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_aperture_sysmem_op_atom_evict_last                           Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-last                                                     
lts__t_sectors_aperture_sysmem_op_atom_evict_last_lookup_hit                Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_aperture_sysmem_op_atom_evict_last_lookup_miss               Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_aperture_sysmem_op_atom_evict_normal                         Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_aperture_sysmem_op_atom_evict_normal_lookup_hit              Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_aperture_sysmem_op_atom_evict_normal_lookup_miss             Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_aperture_sysmem_op_atom_lookup_hit                           Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            that hit                                                              
lts__t_sectors_aperture_sysmem_op_atom_lookup_miss                          Counter         sector          # of LTS sectors accessing system memory (sysmem) for all atomics     
                                                                                                            that missed                                                           
lts__t_sectors_aperture_sysmem_op_membar                                    Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
lts__t_sectors_aperture_sysmem_op_membar_evict_first                        Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-first                                                    
lts__t_sectors_aperture_sysmem_op_membar_evict_first_lookup_hit             Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_aperture_sysmem_op_membar_evict_first_lookup_miss            Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_aperture_sysmem_op_membar_evict_last                         Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-last                                                     
lts__t_sectors_aperture_sysmem_op_membar_evict_last_lookup_hit              Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_aperture_sysmem_op_membar_evict_last_lookup_miss             Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_aperture_sysmem_op_membar_evict_normal                       Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_aperture_sysmem_op_membar_evict_normal_demote                Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_aperture_sysmem_op_membar_evict_normal_demote_lookup_hit     Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_aperture_sysmem_op_membar_evict_normal_demote_lookup_miss    Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_aperture_sysmem_op_membar_evict_normal_lookup_hit            Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_aperture_sysmem_op_membar_evict_normal_lookup_miss           Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_aperture_sysmem_op_membar_lookup_hit                         Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            that hit                                                              
lts__t_sectors_aperture_sysmem_op_membar_lookup_miss                        Counter         sector          # of LTS sectors accessing system memory (sysmem) for memory barriers 
                                                                                                            that missed                                                           
lts__t_sectors_aperture_sysmem_op_read                                      Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads           
lts__t_sectors_aperture_sysmem_op_read_evict_first                          Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-first                                                           
lts__t_sectors_aperture_sysmem_op_read_evict_first_lookup_hit               Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-first that hit                                                  
lts__t_sectors_aperture_sysmem_op_read_evict_first_lookup_miss              Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-first that missed                                               
lts__t_sectors_aperture_sysmem_op_read_evict_last                           Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-last                                                            
lts__t_sectors_aperture_sysmem_op_read_evict_last_lookup_hit                Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-last that hit                                                   
lts__t_sectors_aperture_sysmem_op_read_evict_last_lookup_miss               Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-last that missed                                                
lts__t_sectors_aperture_sysmem_op_read_evict_normal                         Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_aperture_sysmem_op_read_evict_normal_demote                  Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_sysmem_op_read_evict_normal_demote_lookup_hit       Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_sysmem_op_read_evict_normal_demote_lookup_miss      Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_sysmem_op_read_evict_normal_lookup_hit              Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_aperture_sysmem_op_read_evict_normal_lookup_miss             Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads marked    
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_aperture_sysmem_op_read_lookup_hit                           Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads that hit  
lts__t_sectors_aperture_sysmem_op_read_lookup_miss                          Counter         sector          # of LTS sectors accessing system memory (sysmem) for reads that      
                                                                                                            missed                                                                
lts__t_sectors_aperture_sysmem_op_red                                       Counter         sector          # of LTS sectors accessing system memory (sysmem) for reductions      
lts__t_sectors_aperture_sysmem_op_red_lookup_hit                            Counter         sector          # of LTS sectors accessing system memory (sysmem) for reductions that 
                                                                                                            hit                                                                   
lts__t_sectors_aperture_sysmem_op_red_lookup_miss                           Counter         sector          # of LTS sectors accessing system memory (sysmem) for reductions that 
                                                                                                            missed                                                                
lts__t_sectors_aperture_sysmem_op_write                                     Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes          
lts__t_sectors_aperture_sysmem_op_write_evict_first                         Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-first                                                           
lts__t_sectors_aperture_sysmem_op_write_evict_first_lookup_hit              Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-first that hit                                                  
lts__t_sectors_aperture_sysmem_op_write_evict_first_lookup_miss             Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-first that missed                                               
lts__t_sectors_aperture_sysmem_op_write_evict_last                          Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-last                                                            
lts__t_sectors_aperture_sysmem_op_write_evict_last_lookup_hit               Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-last that hit                                                   
lts__t_sectors_aperture_sysmem_op_write_evict_last_lookup_miss              Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-last that missed                                                
lts__t_sectors_aperture_sysmem_op_write_evict_normal                        Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_aperture_sysmem_op_write_evict_normal_demote                 Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_aperture_sysmem_op_write_evict_normal_demote_lookup_hit      Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_aperture_sysmem_op_write_evict_normal_demote_lookup_miss     Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_aperture_sysmem_op_write_evict_normal_lookup_hit             Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_aperture_sysmem_op_write_evict_normal_lookup_miss            Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes marked   
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_aperture_sysmem_op_write_lookup_hit                          Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes that hit 
lts__t_sectors_aperture_sysmem_op_write_lookup_miss                         Counter         sector          # of LTS sectors accessing system memory (sysmem) for writes that     
                                                                                                            missed                                                                
lts__t_sectors_data_ecc                                                     Counter         sector          # of sectors for LTS ECC data                                         
lts__t_sectors_data_user                                                    Counter         sector          # of sectors for LTS user data                                        
lts__t_sectors_equiv_l1tagmiss                                              Counter         sector          # of sectors requested                                                
lts__t_sectors_equiv_l1tagmiss_pipe_lsu                                     Counter         sector          # of sectors requested for LSU pipe local/global                      
lts__t_sectors_equiv_l1tagmiss_pipe_lsu_mem_global_op_atom                  Counter         sector          # of sectors requested for global atomics                             
lts__t_sectors_equiv_l1tagmiss_pipe_lsu_mem_global_op_ld                    Counter         sector          # of sectors requested for global loads                               
lts__t_sectors_equiv_l1tagmiss_pipe_lsu_mem_global_op_red                   Counter         sector          # of sectors requested for global reductions                          
lts__t_sectors_equiv_l1tagmiss_pipe_lsu_mem_global_op_st                    Counter         sector          # of sectors requested for global stores                              
lts__t_sectors_equiv_l1tagmiss_pipe_lsu_mem_local_op_ld                     Counter         sector          # of sectors requested for local loads                                
lts__t_sectors_equiv_l1tagmiss_pipe_lsu_mem_local_op_st                     Counter         sector          # of sectors requested for local stores                               
lts__t_sectors_equiv_l1tagmiss_pipe_tex                                     Counter         sector          # of sectors requested for TEX pipe                                   
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_surface                         Counter         sector          # of sectors requested for surface                                    
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_surface_op_atom                 Counter         sector          # of sectors requested for surface atomics                            
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_surface_op_ld                   Counter         sector          # of sectors requested for surface loads                              
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_surface_op_red                  Counter         sector          # of sectors requested for surface reductions                         
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_surface_op_st                   Counter         sector          # of sectors requested for surface stores                             
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_texture                         Counter         sector          # of sectors requested for texture                                    
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_texture_op_ld                   Counter         sector          # of sectors requested for TLD instructions                           
lts__t_sectors_equiv_l1tagmiss_pipe_tex_mem_texture_op_tex                  Counter         sector          # of sectors requested for TEX instructions                           
lts__t_sectors_evict_first                                                  Counter         sector          # of LTS sectors marked evict-first                                   
lts__t_sectors_evict_first_lookup_hit                                       Counter         sector          # of LTS sectors marked evict-first that hit                          
lts__t_sectors_evict_first_lookup_miss                                      Counter         sector          # of LTS sectors marked evict-first that missed                       
lts__t_sectors_evict_last                                                   Counter         sector          # of LTS sectors marked evict-last                                    
lts__t_sectors_evict_last_lookup_hit                                        Counter         sector          # of LTS sectors marked evict-last that hit                           
lts__t_sectors_evict_last_lookup_miss                                       Counter         sector          # of LTS sectors marked evict-last that missed                        
lts__t_sectors_evict_normal                                                 Counter         sector          # of LTS sectors marked evict-normal (LRU)                            
lts__t_sectors_evict_normal_demote                                          Counter         sector          # of LTS sectors marked evict-normal-demote                           
lts__t_sectors_evict_normal_demote_lookup_hit                               Counter         sector          # of LTS sectors marked evict-normal-demote that hit                  
lts__t_sectors_evict_normal_demote_lookup_miss                              Counter         sector          # of LTS sectors marked evict-normal-demote that missed               
lts__t_sectors_evict_normal_lookup_hit                                      Counter         sector          # of LTS sectors marked evict-normal (LRU) that hit                   
lts__t_sectors_evict_normal_lookup_miss                                     Counter         sector          # of LTS sectors marked evict-normal (LRU) that missed                
lts__t_sectors_lookup_hit                                                   Counter         sector          # of LTS sectors that hit                                             
lts__t_sectors_lookup_miss                                                  Counter         sector          # of LTS sectors that missed                                          
lts__t_sectors_op_atom                                                      Counter         sector          # of LTS sectors for all atomics                                      
lts__t_sectors_op_atom_dot_alu                                              Counter         sector          # of LTS sectors for atomic ALU (non-CAS)                             
lts__t_sectors_op_atom_dot_alu_lookup_hit                                   Counter         sector          # of LTS sectors for atomic ALU (non-CAS) that hit                    
lts__t_sectors_op_atom_dot_alu_lookup_miss                                  Counter         sector          # of LTS sectors for atomic ALU (non-CAS) that missed                 
lts__t_sectors_op_atom_dot_cas                                              Counter         sector          # of LTS sectors for atomic CAS                                       
lts__t_sectors_op_atom_dot_cas_lookup_hit                                   Counter         sector          # of LTS sectors for atomic CAS that hit                              
lts__t_sectors_op_atom_dot_cas_lookup_miss                                  Counter         sector          # of LTS sectors for atomic CAS that missed                           
lts__t_sectors_op_atom_evict_first                                          Counter         sector          # of LTS sectors for all atomics marked evict-first                   
lts__t_sectors_op_atom_evict_first_lookup_hit                               Counter         sector          # of LTS sectors for all atomics marked evict-first that hit          
lts__t_sectors_op_atom_evict_first_lookup_miss                              Counter         sector          # of LTS sectors for all atomics marked evict-first that missed       
lts__t_sectors_op_atom_evict_last                                           Counter         sector          # of LTS sectors for all atomics marked evict-last                    
lts__t_sectors_op_atom_evict_last_lookup_hit                                Counter         sector          # of LTS sectors for all atomics marked evict-last that hit           
lts__t_sectors_op_atom_evict_last_lookup_miss                               Counter         sector          # of LTS sectors for all atomics marked evict-last that missed        
lts__t_sectors_op_atom_evict_normal                                         Counter         sector          # of LTS sectors for all atomics marked evict-normal (LRU)            
lts__t_sectors_op_atom_evict_normal_lookup_hit                              Counter         sector          # of LTS sectors for all atomics marked evict-normal (LRU) that hit   
lts__t_sectors_op_atom_evict_normal_lookup_miss                             Counter         sector          # of LTS sectors for all atomics marked evict-normal (LRU) that missed
lts__t_sectors_op_atom_lookup_hit                                           Counter         sector          # of LTS sectors for all atomics that hit                             
lts__t_sectors_op_atom_lookup_miss                                          Counter         sector          # of LTS sectors for all atomics that missed                          
lts__t_sectors_op_membar                                                    Counter         sector          # of LTS sectors for memory barriers                                  
lts__t_sectors_op_membar_evict_first                                        Counter         sector          # of LTS sectors for memory barriers marked evict-first               
lts__t_sectors_op_membar_evict_first_lookup_hit                             Counter         sector          # of LTS sectors for memory barriers marked evict-first that hit      
lts__t_sectors_op_membar_evict_first_lookup_miss                            Counter         sector          # of LTS sectors for memory barriers marked evict-first that missed   
lts__t_sectors_op_membar_evict_last                                         Counter         sector          # of LTS sectors for memory barriers marked evict-last                
lts__t_sectors_op_membar_evict_last_lookup_hit                              Counter         sector          # of LTS sectors for memory barriers marked evict-last that hit       
lts__t_sectors_op_membar_evict_last_lookup_miss                             Counter         sector          # of LTS sectors for memory barriers marked evict-last that missed    
lts__t_sectors_op_membar_evict_normal                                       Counter         sector          # of LTS sectors for memory barriers marked evict-normal (LRU)        
lts__t_sectors_op_membar_evict_normal_demote                                Counter         sector          # of LTS sectors for memory barriers marked evict-normal-demote       
lts__t_sectors_op_membar_evict_normal_demote_lookup_hit                     Counter         sector          # of LTS sectors for memory barriers marked evict-normal-demote that  
                                                                                                            hit                                                                   
lts__t_sectors_op_membar_evict_normal_demote_lookup_miss                    Counter         sector          # of LTS sectors for memory barriers marked evict-normal-demote that  
                                                                                                            missed                                                                
lts__t_sectors_op_membar_evict_normal_lookup_hit                            Counter         sector          # of LTS sectors for memory barriers marked evict-normal (LRU) that   
                                                                                                            hit                                                                   
lts__t_sectors_op_membar_evict_normal_lookup_miss                           Counter         sector          # of LTS sectors for memory barriers marked evict-normal (LRU) that   
                                                                                                            missed                                                                
lts__t_sectors_op_membar_lookup_hit                                         Counter         sector          # of LTS sectors for memory barriers that hit                         
lts__t_sectors_op_membar_lookup_miss                                        Counter         sector          # of LTS sectors for memory barriers that missed                      
lts__t_sectors_op_read                                                      Counter         sector          # of LTS sectors for reads                                            
lts__t_sectors_op_read_evict_first                                          Counter         sector          # of LTS sectors for reads marked evict-first                         
lts__t_sectors_op_read_evict_first_lookup_hit                               Counter         sector          # of LTS sectors for reads marked evict-first that hit                
lts__t_sectors_op_read_evict_first_lookup_miss                              Counter         sector          # of LTS sectors for reads marked evict-first that missed             
lts__t_sectors_op_read_evict_last                                           Counter         sector          # of LTS sectors for reads marked evict-last                          
lts__t_sectors_op_read_evict_last_lookup_hit                                Counter         sector          # of LTS sectors for reads marked evict-last that hit                 
lts__t_sectors_op_read_evict_last_lookup_miss                               Counter         sector          # of LTS sectors for reads marked evict-last that missed              
lts__t_sectors_op_read_evict_normal                                         Counter         sector          # of LTS sectors for reads marked evict-normal (LRU)                  
lts__t_sectors_op_read_evict_normal_demote                                  Counter         sector          # of LTS sectors for reads marked evict-normal-demote                 
lts__t_sectors_op_read_evict_normal_demote_lookup_hit                       Counter         sector          # of LTS sectors for reads marked evict-normal-demote that hit        
lts__t_sectors_op_read_evict_normal_demote_lookup_miss                      Counter         sector          # of LTS sectors for reads marked evict-normal-demote that missed     
lts__t_sectors_op_read_evict_normal_lookup_hit                              Counter         sector          # of LTS sectors for reads marked evict-normal (LRU) that hit         
lts__t_sectors_op_read_evict_normal_lookup_miss                             Counter         sector          # of LTS sectors for reads marked evict-normal (LRU) that missed      
lts__t_sectors_op_read_lookup_hit                                           Counter         sector          # of LTS sectors for reads that hit                                   
lts__t_sectors_op_read_lookup_miss                                          Counter         sector          # of LTS sectors for reads that missed                                
lts__t_sectors_op_red                                                       Counter         sector          # of LTS sectors for reductions                                       
lts__t_sectors_op_red_lookup_hit                                            Counter         sector          # of LTS sectors for reductions that hit                              
lts__t_sectors_op_red_lookup_miss                                           Counter         sector          # of LTS sectors for reductions that missed                           
lts__t_sectors_op_write                                                     Counter         sector          # of LTS sectors for writes                                           
lts__t_sectors_op_write_evict_first                                         Counter         sector          # of LTS sectors for writes marked evict-first                        
lts__t_sectors_op_write_evict_first_lookup_hit                              Counter         sector          # of LTS sectors for writes marked evict-first that hit               
lts__t_sectors_op_write_evict_first_lookup_miss                             Counter         sector          # of LTS sectors for writes marked evict-first that missed            
lts__t_sectors_op_write_evict_last                                          Counter         sector          # of LTS sectors for writes marked evict-last                         
lts__t_sectors_op_write_evict_last_lookup_hit                               Counter         sector          # of LTS sectors for writes marked evict-last that hit                
lts__t_sectors_op_write_evict_last_lookup_miss                              Counter         sector          # of LTS sectors for writes marked evict-last that missed             
lts__t_sectors_op_write_evict_normal                                        Counter         sector          # of LTS sectors for writes marked evict-normal (LRU)                 
lts__t_sectors_op_write_evict_normal_demote                                 Counter         sector          # of LTS sectors for writes marked evict-normal-demote                
lts__t_sectors_op_write_evict_normal_demote_lookup_hit                      Counter         sector          # of LTS sectors for writes marked evict-normal-demote that hit       
lts__t_sectors_op_write_evict_normal_demote_lookup_miss                     Counter         sector          # of LTS sectors for writes marked evict-normal-demote that missed    
lts__t_sectors_op_write_evict_normal_lookup_hit                             Counter         sector          # of LTS sectors for writes marked evict-normal (LRU) that hit        
lts__t_sectors_op_write_evict_normal_lookup_miss                            Counter         sector          # of LTS sectors for writes marked evict-normal (LRU) that missed     
lts__t_sectors_op_write_lookup_hit                                          Counter         sector          # of LTS sectors for writes that hit                                  
lts__t_sectors_op_write_lookup_miss                                         Counter         sector          # of LTS sectors for writes that missed                               
lts__t_sectors_srcnode_gpc                                                  Counter         sector          # of LTS sectors from node GPC                                        
lts__t_sectors_srcnode_gpc_aperture_device                                  Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
lts__t_sectors_srcnode_gpc_aperture_device_evict_first                      Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-first                                                    
lts__t_sectors_srcnode_gpc_aperture_device_evict_first_lookup_hit           Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_srcnode_gpc_aperture_device_evict_first_lookup_miss          Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_srcnode_gpc_aperture_device_evict_last                       Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-last                                                     
lts__t_sectors_srcnode_gpc_aperture_device_evict_last_lookup_hit            Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_srcnode_gpc_aperture_device_evict_last_lookup_miss           Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_srcnode_gpc_aperture_device_evict_normal                     Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_srcnode_gpc_aperture_device_evict_normal_demote              Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_srcnode_gpc_aperture_device_evict_normal_demote_lookup_hit   Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_srcnode_gpc_aperture_device_evict_normal_demote_lookup_miss  Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_srcnode_gpc_aperture_device_evict_normal_lookup_hit          Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcnode_gpc_aperture_device_evict_normal_lookup_miss         Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcnode_gpc_aperture_device_lookup_hit                       Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) that  
                                                                                                            hit                                                                   
lts__t_sectors_srcnode_gpc_aperture_device_lookup_miss                      Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) that  
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_aperture_device_op_atom                          Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            all atomics                                                           
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_dot_alu                  Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_dot_alu_lookup_hit       Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_dot_alu_lookup_miss      Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_dot_cas                  Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_dot_cas_lookup_hit       Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_dot_cas_lookup_miss      Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_lookup_hit               Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcnode_gpc_aperture_device_op_atom_lookup_miss              Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcnode_gpc_aperture_device_op_membar                        Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            memory barriers                                                       
lts__t_sectors_srcnode_gpc_aperture_device_op_membar_lookup_hit             Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcnode_gpc_aperture_device_op_membar_lookup_miss            Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcnode_gpc_aperture_device_op_read                          Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            reads                                                                 
lts__t_sectors_srcnode_gpc_aperture_device_op_read_lookup_hit               Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            reads that hit                                                        
lts__t_sectors_srcnode_gpc_aperture_device_op_read_lookup_miss              Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            reads that missed                                                     
lts__t_sectors_srcnode_gpc_aperture_device_op_red                           Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            reductions                                                            
lts__t_sectors_srcnode_gpc_aperture_device_op_red_lookup_hit                Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_device_op_red_lookup_miss               Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            reductions that missed                                                
lts__t_sectors_srcnode_gpc_aperture_device_op_write                         Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            writes                                                                
lts__t_sectors_srcnode_gpc_aperture_device_op_write_lookup_hit              Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            writes that hit                                                       
lts__t_sectors_srcnode_gpc_aperture_device_op_write_lookup_miss             Counter         sector          # of LTS sectors from node GPC accessing device memory (vidmem) for   
                                                                                                            writes that missed                                                    
lts__t_sectors_srcnode_gpc_aperture_peer                                    Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem)        
lts__t_sectors_srcnode_gpc_aperture_peer_evict_first                        Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-first                                                           
lts__t_sectors_srcnode_gpc_aperture_peer_evict_first_lookup_hit             Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-first that hit                                                  
lts__t_sectors_srcnode_gpc_aperture_peer_evict_first_lookup_miss            Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-first that missed                                               
lts__t_sectors_srcnode_gpc_aperture_peer_evict_last                         Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-last                                                            
lts__t_sectors_srcnode_gpc_aperture_peer_evict_last_lookup_hit              Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-last that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_peer_evict_last_lookup_miss             Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-last that missed                                                
lts__t_sectors_srcnode_gpc_aperture_peer_evict_normal                       Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcnode_gpc_aperture_peer_evict_normal_demote                Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcnode_gpc_aperture_peer_evict_normal_demote_lookup_hit     Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcnode_gpc_aperture_peer_evict_normal_demote_lookup_miss    Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcnode_gpc_aperture_peer_evict_normal_lookup_hit            Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcnode_gpc_aperture_peer_evict_normal_lookup_miss           Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcnode_gpc_aperture_peer_lookup_hit                         Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) that   
                                                                                                            hit                                                                   
lts__t_sectors_srcnode_gpc_aperture_peer_lookup_miss                        Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) that   
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom                            Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            all atomics                                                           
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_dot_alu                    Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_dot_alu_lookup_hit         Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_dot_alu_lookup_miss        Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_dot_cas                    Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_dot_cas_lookup_hit         Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_dot_cas_lookup_miss        Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_lookup_hit                 Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcnode_gpc_aperture_peer_op_atom_lookup_miss                Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcnode_gpc_aperture_peer_op_membar                          Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            memory barriers                                                       
lts__t_sectors_srcnode_gpc_aperture_peer_op_membar_lookup_hit               Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcnode_gpc_aperture_peer_op_membar_lookup_miss              Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcnode_gpc_aperture_peer_op_read                            Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            reads                                                                 
lts__t_sectors_srcnode_gpc_aperture_peer_op_read_lookup_hit                 Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            reads that hit                                                        
lts__t_sectors_srcnode_gpc_aperture_peer_op_read_lookup_miss                Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            reads that missed                                                     
lts__t_sectors_srcnode_gpc_aperture_peer_op_red                             Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            reductions                                                            
lts__t_sectors_srcnode_gpc_aperture_peer_op_red_lookup_hit                  Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_peer_op_red_lookup_miss                 Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            reductions that missed                                                
lts__t_sectors_srcnode_gpc_aperture_peer_op_write                           Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            writes                                                                
lts__t_sectors_srcnode_gpc_aperture_peer_op_write_lookup_hit                Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            writes that hit                                                       
lts__t_sectors_srcnode_gpc_aperture_peer_op_write_lookup_miss               Counter         sector          # of LTS sectors from node GPC accessing peer memory (peermem) for    
                                                                                                            writes that missed                                                    
lts__t_sectors_srcnode_gpc_aperture_sysmem                                  Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_first                      Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-first                                                    
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_first_lookup_hit           Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_first_lookup_miss          Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_last                       Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-last                                                     
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_last_lookup_hit            Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_last_lookup_miss           Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_normal                     Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_normal_demote              Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_normal_demote_lookup_hit   Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_normal_demote_lookup_miss  Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_normal_lookup_hit          Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcnode_gpc_aperture_sysmem_evict_normal_lookup_miss         Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcnode_gpc_aperture_sysmem_lookup_hit                       Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) that  
                                                                                                            hit                                                                   
lts__t_sectors_srcnode_gpc_aperture_sysmem_lookup_miss                      Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) that  
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom                          Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            all atomics                                                           
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_dot_alu                  Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_dot_alu_lookup_hit       Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_dot_alu_lookup_miss      Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_dot_cas                  Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_dot_cas_lookup_hit       Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_dot_cas_lookup_miss      Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_lookup_hit               Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_atom_lookup_miss              Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_membar                        Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            memory barriers                                                       
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_membar_lookup_hit             Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_membar_lookup_miss            Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_read                          Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            reads                                                                 
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_read_lookup_hit               Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            reads that hit                                                        
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_read_lookup_miss              Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            reads that missed                                                     
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_red                           Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            reductions                                                            
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_red_lookup_hit                Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_red_lookup_miss               Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            reductions that missed                                                
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_write                         Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            writes                                                                
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_write_lookup_hit              Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            writes that hit                                                       
lts__t_sectors_srcnode_gpc_aperture_sysmem_op_write_lookup_miss             Counter         sector          # of LTS sectors from node GPC accessing system memory (sysmem) for   
                                                                                                            writes that missed                                                    
lts__t_sectors_srcnode_gpc_evict_first                                      Counter         sector          # of LTS sectors from node GPC marked evict-first                     
lts__t_sectors_srcnode_gpc_evict_first_lookup_hit                           Counter         sector          # of LTS sectors from node GPC marked evict-first that hit            
lts__t_sectors_srcnode_gpc_evict_first_lookup_miss                          Counter         sector          # of LTS sectors from node GPC marked evict-first that missed         
lts__t_sectors_srcnode_gpc_evict_last                                       Counter         sector          # of LTS sectors from node GPC marked evict-last                      
lts__t_sectors_srcnode_gpc_evict_last_lookup_hit                            Counter         sector          # of LTS sectors from node GPC marked evict-last that hit             
lts__t_sectors_srcnode_gpc_evict_last_lookup_miss                           Counter         sector          # of LTS sectors from node GPC marked evict-last that missed          
lts__t_sectors_srcnode_gpc_evict_normal                                     Counter         sector          # of LTS sectors from node GPC marked evict-normal (LRU)              
lts__t_sectors_srcnode_gpc_evict_normal_demote                              Counter         sector          # of LTS sectors from node GPC marked evict-normal-demote             
lts__t_sectors_srcnode_gpc_evict_normal_demote_lookup_hit                   Counter         sector          # of LTS sectors from node GPC marked evict-normal-demote that hit    
lts__t_sectors_srcnode_gpc_evict_normal_demote_lookup_miss                  Counter         sector          # of LTS sectors from node GPC marked evict-normal-demote that missed 
lts__t_sectors_srcnode_gpc_evict_normal_lookup_hit                          Counter         sector          # of LTS sectors from node GPC marked evict-normal (LRU) that hit     
lts__t_sectors_srcnode_gpc_evict_normal_lookup_miss                         Counter         sector          # of LTS sectors from node GPC marked evict-normal (LRU) that missed  
lts__t_sectors_srcnode_gpc_lookup_hit                                       Counter         sector          # of LTS sectors from node GPC that hit                               
lts__t_sectors_srcnode_gpc_lookup_miss                                      Counter         sector          # of LTS sectors from node GPC that missed                            
lts__t_sectors_srcnode_gpc_op_atom                                          Counter         sector          # of LTS sectors from node GPC for all atomics                        
lts__t_sectors_srcnode_gpc_op_atom_dot_alu                                  Counter         sector          # of LTS sectors from node GPC for atomic ALU (non-CAS)               
lts__t_sectors_srcnode_gpc_op_atom_dot_alu_lookup_hit                       Counter         sector          # of LTS sectors from node GPC for atomic ALU (non-CAS) that hit      
lts__t_sectors_srcnode_gpc_op_atom_dot_alu_lookup_miss                      Counter         sector          # of LTS sectors from node GPC for atomic ALU (non-CAS) that missed   
lts__t_sectors_srcnode_gpc_op_atom_dot_cas                                  Counter         sector          # of LTS sectors from node GPC for atomic CAS                         
lts__t_sectors_srcnode_gpc_op_atom_dot_cas_lookup_hit                       Counter         sector          # of LTS sectors from node GPC for atomic CAS that hit                
lts__t_sectors_srcnode_gpc_op_atom_dot_cas_lookup_miss                      Counter         sector          # of LTS sectors from node GPC for atomic CAS that missed             
lts__t_sectors_srcnode_gpc_op_atom_evict_first                              Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-first     
lts__t_sectors_srcnode_gpc_op_atom_evict_first_lookup_hit                   Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-first     
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_atom_evict_first_lookup_miss                  Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-first     
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_atom_evict_last                               Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-last      
lts__t_sectors_srcnode_gpc_op_atom_evict_last_lookup_hit                    Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-last that 
                                                                                                            hit                                                                   
lts__t_sectors_srcnode_gpc_op_atom_evict_last_lookup_miss                   Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-last that 
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_op_atom_evict_normal                             Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-normal    
                                                                                                            (LRU)                                                                 
lts__t_sectors_srcnode_gpc_op_atom_evict_normal_lookup_hit                  Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-normal    
                                                                                                            (LRU) that hit                                                        
lts__t_sectors_srcnode_gpc_op_atom_evict_normal_lookup_miss                 Counter         sector          # of LTS sectors from node GPC for all atomics marked evict-normal    
                                                                                                            (LRU) that missed                                                     
lts__t_sectors_srcnode_gpc_op_atom_lookup_hit                               Counter         sector          # of LTS sectors from node GPC for all atomics that hit               
lts__t_sectors_srcnode_gpc_op_atom_lookup_miss                              Counter         sector          # of LTS sectors from node GPC for all atomics that missed            
lts__t_sectors_srcnode_gpc_op_membar                                        Counter         sector          # of LTS sectors from node GPC for memory barriers                    
lts__t_sectors_srcnode_gpc_op_membar_evict_first                            Counter         sector          # of LTS sectors from node GPC for memory barriers marked evict-first 
lts__t_sectors_srcnode_gpc_op_membar_evict_first_lookup_hit                 Counter         sector          # of LTS sectors from node GPC for memory barriers marked evict-first 
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_membar_evict_first_lookup_miss                Counter         sector          # of LTS sectors from node GPC for memory barriers marked evict-first 
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_membar_evict_last                             Counter         sector          # of LTS sectors from node GPC for memory barriers marked evict-last  
lts__t_sectors_srcnode_gpc_op_membar_evict_last_lookup_hit                  Counter         sector          # of LTS sectors from node GPC for memory barriers marked evict-last  
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_membar_evict_last_lookup_miss                 Counter         sector          # of LTS sectors from node GPC for memory barriers marked evict-last  
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_membar_evict_normal                           Counter         sector          # of LTS sectors from node GPC for memory barriers marked             
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcnode_gpc_op_membar_evict_normal_demote                    Counter         sector          # of LTS sectors from node GPC for memory barriers marked             
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcnode_gpc_op_membar_evict_normal_demote_lookup_hit         Counter         sector          # of LTS sectors from node GPC for memory barriers marked             
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcnode_gpc_op_membar_evict_normal_demote_lookup_miss        Counter         sector          # of LTS sectors from node GPC for memory barriers marked             
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcnode_gpc_op_membar_evict_normal_lookup_hit                Counter         sector          # of LTS sectors from node GPC for memory barriers marked             
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcnode_gpc_op_membar_evict_normal_lookup_miss               Counter         sector          # of LTS sectors from node GPC for memory barriers marked             
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcnode_gpc_op_membar_lookup_hit                             Counter         sector          # of LTS sectors from node GPC for memory barriers that hit           
lts__t_sectors_srcnode_gpc_op_membar_lookup_miss                            Counter         sector          # of LTS sectors from node GPC for memory barriers that missed        
lts__t_sectors_srcnode_gpc_op_read                                          Counter         sector          # of LTS sectors from node GPC for reads                              
lts__t_sectors_srcnode_gpc_op_read_evict_first                              Counter         sector          # of LTS sectors from node GPC for reads marked evict-first           
lts__t_sectors_srcnode_gpc_op_read_evict_first_lookup_hit                   Counter         sector          # of LTS sectors from node GPC for reads marked evict-first that hit  
lts__t_sectors_srcnode_gpc_op_read_evict_first_lookup_miss                  Counter         sector          # of LTS sectors from node GPC for reads marked evict-first that      
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_op_read_evict_last                               Counter         sector          # of LTS sectors from node GPC for reads marked evict-last            
lts__t_sectors_srcnode_gpc_op_read_evict_last_lookup_hit                    Counter         sector          # of LTS sectors from node GPC for reads marked evict-last that hit   
lts__t_sectors_srcnode_gpc_op_read_evict_last_lookup_miss                   Counter         sector          # of LTS sectors from node GPC for reads marked evict-last that missed
lts__t_sectors_srcnode_gpc_op_read_evict_normal                             Counter         sector          # of LTS sectors from node GPC for reads marked evict-normal (LRU)    
lts__t_sectors_srcnode_gpc_op_read_evict_normal_demote                      Counter         sector          # of LTS sectors from node GPC for reads marked evict-normal-demote   
lts__t_sectors_srcnode_gpc_op_read_evict_normal_demote_lookup_hit           Counter         sector          # of LTS sectors from node GPC for reads marked evict-normal-demote   
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_read_evict_normal_demote_lookup_miss          Counter         sector          # of LTS sectors from node GPC for reads marked evict-normal-demote   
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_read_evict_normal_lookup_hit                  Counter         sector          # of LTS sectors from node GPC for reads marked evict-normal (LRU)    
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_read_evict_normal_lookup_miss                 Counter         sector          # of LTS sectors from node GPC for reads marked evict-normal (LRU)    
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_read_lookup_hit                               Counter         sector          # of LTS sectors from node GPC for reads that hit                     
lts__t_sectors_srcnode_gpc_op_read_lookup_miss                              Counter         sector          # of LTS sectors from node GPC for reads that missed                  
lts__t_sectors_srcnode_gpc_op_red                                           Counter         sector          # of LTS sectors from node GPC for reductions                         
lts__t_sectors_srcnode_gpc_op_red_lookup_hit                                Counter         sector          # of LTS sectors from node GPC for reductions that hit                
lts__t_sectors_srcnode_gpc_op_red_lookup_miss                               Counter         sector          # of LTS sectors from node GPC for reductions that missed             
lts__t_sectors_srcnode_gpc_op_write                                         Counter         sector          # of LTS sectors from node GPC for writes                             
lts__t_sectors_srcnode_gpc_op_write_evict_first                             Counter         sector          # of LTS sectors from node GPC for writes marked evict-first          
lts__t_sectors_srcnode_gpc_op_write_evict_first_lookup_hit                  Counter         sector          # of LTS sectors from node GPC for writes marked evict-first that hit 
lts__t_sectors_srcnode_gpc_op_write_evict_first_lookup_miss                 Counter         sector          # of LTS sectors from node GPC for writes marked evict-first that     
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_op_write_evict_last                              Counter         sector          # of LTS sectors from node GPC for writes marked evict-last           
lts__t_sectors_srcnode_gpc_op_write_evict_last_lookup_hit                   Counter         sector          # of LTS sectors from node GPC for writes marked evict-last that hit  
lts__t_sectors_srcnode_gpc_op_write_evict_last_lookup_miss                  Counter         sector          # of LTS sectors from node GPC for writes marked evict-last that      
                                                                                                            missed                                                                
lts__t_sectors_srcnode_gpc_op_write_evict_normal                            Counter         sector          # of LTS sectors from node GPC for writes marked evict-normal (LRU)   
lts__t_sectors_srcnode_gpc_op_write_evict_normal_demote                     Counter         sector          # of LTS sectors from node GPC for writes marked evict-normal-demote  
lts__t_sectors_srcnode_gpc_op_write_evict_normal_demote_lookup_hit          Counter         sector          # of LTS sectors from node GPC for writes marked evict-normal-demote  
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_write_evict_normal_demote_lookup_miss         Counter         sector          # of LTS sectors from node GPC for writes marked evict-normal-demote  
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_write_evict_normal_lookup_hit                 Counter         sector          # of LTS sectors from node GPC for writes marked evict-normal (LRU)   
                                                                                                            that hit                                                              
lts__t_sectors_srcnode_gpc_op_write_evict_normal_lookup_miss                Counter         sector          # of LTS sectors from node GPC for writes marked evict-normal (LRU)   
                                                                                                            that missed                                                           
lts__t_sectors_srcnode_gpc_op_write_lookup_hit                              Counter         sector          # of LTS sectors from node GPC for writes that hit                    
lts__t_sectors_srcnode_gpc_op_write_lookup_miss                             Counter         sector          # of LTS sectors from node GPC for writes that missed                 
lts__t_sectors_srcunit_l1                                                   Counter         sector          # of LTS sectors from unit L1                                         
lts__t_sectors_srcunit_l1_aperture_device                                   Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem)        
lts__t_sectors_srcunit_l1_aperture_device_evict_first                       Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-first                                                           
lts__t_sectors_srcunit_l1_aperture_device_evict_first_lookup_hit            Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-first that hit                                                  
lts__t_sectors_srcunit_l1_aperture_device_evict_first_lookup_miss           Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-first that missed                                               
lts__t_sectors_srcunit_l1_aperture_device_evict_last                        Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-last                                                            
lts__t_sectors_srcunit_l1_aperture_device_evict_last_lookup_hit             Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-last that hit                                                   
lts__t_sectors_srcunit_l1_aperture_device_evict_last_lookup_miss            Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-last that missed                                                
lts__t_sectors_srcunit_l1_aperture_device_evict_normal                      Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcunit_l1_aperture_device_evict_normal_demote               Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcunit_l1_aperture_device_evict_normal_demote_lookup_hit    Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcunit_l1_aperture_device_evict_normal_demote_lookup_miss   Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcunit_l1_aperture_device_evict_normal_lookup_hit           Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcunit_l1_aperture_device_evict_normal_lookup_miss          Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) marked 
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcunit_l1_aperture_device_lookup_hit                        Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) that   
                                                                                                            hit                                                                   
lts__t_sectors_srcunit_l1_aperture_device_lookup_miss                       Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) that   
                                                                                                            missed                                                                
lts__t_sectors_srcunit_l1_aperture_device_op_atom                           Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            all atomics                                                           
lts__t_sectors_srcunit_l1_aperture_device_op_atom_dot_alu                   Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcunit_l1_aperture_device_op_atom_dot_alu_lookup_hit        Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcunit_l1_aperture_device_op_atom_dot_alu_lookup_miss       Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcunit_l1_aperture_device_op_atom_dot_cas                   Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcunit_l1_aperture_device_op_atom_dot_cas_lookup_hit        Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcunit_l1_aperture_device_op_atom_dot_cas_lookup_miss       Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcunit_l1_aperture_device_op_atom_lookup_hit                Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcunit_l1_aperture_device_op_atom_lookup_miss               Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcunit_l1_aperture_device_op_membar                         Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            memory barriers                                                       
lts__t_sectors_srcunit_l1_aperture_device_op_membar_lookup_hit              Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcunit_l1_aperture_device_op_membar_lookup_miss             Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcunit_l1_aperture_device_op_read                           Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            reads                                                                 
lts__t_sectors_srcunit_l1_aperture_device_op_read_lookup_hit                Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            reads that hit                                                        
lts__t_sectors_srcunit_l1_aperture_device_op_read_lookup_miss               Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            reads that missed                                                     
lts__t_sectors_srcunit_l1_aperture_device_op_red                            Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            reductions                                                            
lts__t_sectors_srcunit_l1_aperture_device_op_red_lookup_hit                 Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcunit_l1_aperture_device_op_red_lookup_miss                Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            reductions that missed                                                
lts__t_sectors_srcunit_l1_aperture_device_op_write                          Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            writes                                                                
lts__t_sectors_srcunit_l1_aperture_device_op_write_lookup_hit               Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            writes that hit                                                       
lts__t_sectors_srcunit_l1_aperture_device_op_write_lookup_miss              Counter         sector          # of LTS sectors from unit L1 accessing device memory (vidmem) for    
                                                                                                            writes that missed                                                    
lts__t_sectors_srcunit_l1_aperture_peer                                     Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem)         
lts__t_sectors_srcunit_l1_aperture_peer_evict_first                         Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-first                                                           
lts__t_sectors_srcunit_l1_aperture_peer_evict_first_lookup_hit              Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-first that hit                                                  
lts__t_sectors_srcunit_l1_aperture_peer_evict_first_lookup_miss             Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-first that missed                                               
lts__t_sectors_srcunit_l1_aperture_peer_evict_last                          Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-last                                                            
lts__t_sectors_srcunit_l1_aperture_peer_evict_last_lookup_hit               Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-last that hit                                                   
lts__t_sectors_srcunit_l1_aperture_peer_evict_last_lookup_miss              Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-last that missed                                                
lts__t_sectors_srcunit_l1_aperture_peer_evict_normal                        Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcunit_l1_aperture_peer_evict_normal_demote                 Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcunit_l1_aperture_peer_evict_normal_demote_lookup_hit      Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcunit_l1_aperture_peer_evict_normal_demote_lookup_miss     Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcunit_l1_aperture_peer_evict_normal_lookup_hit             Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcunit_l1_aperture_peer_evict_normal_lookup_miss            Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) marked  
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcunit_l1_aperture_peer_lookup_hit                          Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) that hit
lts__t_sectors_srcunit_l1_aperture_peer_lookup_miss                         Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) that    
                                                                                                            missed                                                                
lts__t_sectors_srcunit_l1_aperture_peer_op_atom                             Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for all 
                                                                                                            atomics                                                               
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_dot_alu                     Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_dot_alu_lookup_hit          Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_dot_alu_lookup_miss         Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_dot_cas                     Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_dot_cas_lookup_hit          Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_dot_cas_lookup_miss         Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_lookup_hit                  Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for all 
                                                                                                            atomics that hit                                                      
lts__t_sectors_srcunit_l1_aperture_peer_op_atom_lookup_miss                 Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for all 
                                                                                                            atomics that missed                                                   
lts__t_sectors_srcunit_l1_aperture_peer_op_membar                           Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            memory barriers                                                       
lts__t_sectors_srcunit_l1_aperture_peer_op_membar_lookup_hit                Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcunit_l1_aperture_peer_op_membar_lookup_miss               Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcunit_l1_aperture_peer_op_read                             Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            reads                                                                 
lts__t_sectors_srcunit_l1_aperture_peer_op_read_lookup_hit                  Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            reads that hit                                                        
lts__t_sectors_srcunit_l1_aperture_peer_op_read_lookup_miss                 Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            reads that missed                                                     
lts__t_sectors_srcunit_l1_aperture_peer_op_red                              Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            reductions                                                            
lts__t_sectors_srcunit_l1_aperture_peer_op_red_lookup_hit                   Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcunit_l1_aperture_peer_op_red_lookup_miss                  Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            reductions that missed                                                
lts__t_sectors_srcunit_l1_aperture_peer_op_write                            Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            writes                                                                
lts__t_sectors_srcunit_l1_aperture_peer_op_write_lookup_hit                 Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            writes that hit                                                       
lts__t_sectors_srcunit_l1_aperture_peer_op_write_lookup_miss                Counter         sector          # of LTS sectors from unit L1 accessing peer memory (peermem) for     
                                                                                                            writes that missed                                                    
lts__t_sectors_srcunit_l1_aperture_sysmem                                   Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem)        
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_first                       Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-first                                                           
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_first_lookup_hit            Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-first that hit                                                  
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_first_lookup_miss           Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-first that missed                                               
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_last                        Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-last                                                            
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_last_lookup_hit             Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-last that hit                                                   
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_last_lookup_miss            Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-last that missed                                                
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_normal                      Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_normal_demote               Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_normal_demote_lookup_hit    Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_normal_demote_lookup_miss   Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_normal_lookup_hit           Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcunit_l1_aperture_sysmem_evict_normal_lookup_miss          Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) marked 
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcunit_l1_aperture_sysmem_lookup_hit                        Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) that   
                                                                                                            hit                                                                   
lts__t_sectors_srcunit_l1_aperture_sysmem_lookup_miss                       Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) that   
                                                                                                            missed                                                                
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom                           Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            all atomics                                                           
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_dot_alu                   Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_dot_alu_lookup_hit        Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_dot_alu_lookup_miss       Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_dot_cas                   Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_dot_cas_lookup_hit        Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_dot_cas_lookup_miss       Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_lookup_hit                Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcunit_l1_aperture_sysmem_op_atom_lookup_miss               Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcunit_l1_aperture_sysmem_op_membar                         Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            memory barriers                                                       
lts__t_sectors_srcunit_l1_aperture_sysmem_op_membar_lookup_hit              Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcunit_l1_aperture_sysmem_op_membar_lookup_miss             Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcunit_l1_aperture_sysmem_op_read                           Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            reads                                                                 
lts__t_sectors_srcunit_l1_aperture_sysmem_op_read_lookup_hit                Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            reads that hit                                                        
lts__t_sectors_srcunit_l1_aperture_sysmem_op_read_lookup_miss               Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            reads that missed                                                     
lts__t_sectors_srcunit_l1_aperture_sysmem_op_red                            Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            reductions                                                            
lts__t_sectors_srcunit_l1_aperture_sysmem_op_red_lookup_hit                 Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcunit_l1_aperture_sysmem_op_red_lookup_miss                Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            reductions that missed                                                
lts__t_sectors_srcunit_l1_aperture_sysmem_op_write                          Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            writes                                                                
lts__t_sectors_srcunit_l1_aperture_sysmem_op_write_lookup_hit               Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            writes that hit                                                       
lts__t_sectors_srcunit_l1_aperture_sysmem_op_write_lookup_miss              Counter         sector          # of LTS sectors from unit L1 accessing system memory (sysmem) for    
                                                                                                            writes that missed                                                    
lts__t_sectors_srcunit_l1_evict_first                                       Counter         sector          # of LTS sectors from unit L1 marked evict-first                      
lts__t_sectors_srcunit_l1_evict_first_lookup_hit                            Counter         sector          # of LTS sectors from unit L1 marked evict-first that hit             
lts__t_sectors_srcunit_l1_evict_first_lookup_miss                           Counter         sector          # of LTS sectors from unit L1 marked evict-first that missed          
lts__t_sectors_srcunit_l1_evict_last                                        Counter         sector          # of LTS sectors from unit L1 marked evict-last                       
lts__t_sectors_srcunit_l1_evict_last_lookup_hit                             Counter         sector          # of LTS sectors from unit L1 marked evict-last that hit              
lts__t_sectors_srcunit_l1_evict_last_lookup_miss                            Counter         sector          # of LTS sectors from unit L1 marked evict-last that missed           
lts__t_sectors_srcunit_l1_evict_normal                                      Counter         sector          # of LTS sectors from unit L1 marked evict-normal (LRU)               
lts__t_sectors_srcunit_l1_evict_normal_demote                                    Counter         sector          # of LTS sectors from unit L1 marked evict-normal-demote              
lts__t_sectors_srcunit_l1_evict_normal_demote_lookup_hit                         Counter         sector          # of LTS sectors from unit L1 marked evict-normal-demote that hit     
lts__t_sectors_srcunit_l1_evict_normal_demote_lookup_miss                        Counter         sector          # of LTS sectors from unit L1 marked evict-normal-demote that missed  
lts__t_sectors_srcunit_l1_evict_normal_lookup_hit                                Counter         sector          # of LTS sectors from unit L1 marked evict-normal (LRU) that hit      
lts__t_sectors_srcunit_l1_evict_normal_lookup_miss                               Counter         sector          # of LTS sectors from unit L1 marked evict-normal (LRU) that missed   
lts__t_sectors_srcunit_l1_lookup_hit                                             Counter         sector          # of LTS sectors from unit L1 that hit                                
lts__t_sectors_srcunit_l1_lookup_miss                                            Counter         sector          # of LTS sectors from unit L1 that missed                             
lts__t_sectors_srcunit_l1_op_atom                                                Counter         sector          # of LTS sectors from unit L1 for all atomics                         
lts__t_sectors_srcunit_l1_op_atom_dot_alu                                        Counter         sector          # of LTS sectors from unit L1 for atomic ALU (non-CAS)                
lts__t_sectors_srcunit_l1_op_atom_dot_alu_lookup_hit                             Counter         sector          # of LTS sectors from unit L1 for atomic ALU (non-CAS) that hit       
lts__t_sectors_srcunit_l1_op_atom_dot_alu_lookup_miss                            Counter         sector          # of LTS sectors from unit L1 for atomic ALU (non-CAS) that missed    
lts__t_sectors_srcunit_l1_op_atom_dot_cas                                        Counter         sector          # of LTS sectors from unit L1 for atomic CAS                          
lts__t_sectors_srcunit_l1_op_atom_dot_cas_lookup_hit                             Counter         sector          # of LTS sectors from unit L1 for atomic CAS that hit                 
lts__t_sectors_srcunit_l1_op_atom_dot_cas_lookup_miss                            Counter         sector          # of LTS sectors from unit L1 for atomic CAS that missed              
lts__t_sectors_srcunit_l1_op_atom_evict_first                                    Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-first      
lts__t_sectors_srcunit_l1_op_atom_evict_first_lookup_hit                         Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-first that 
                                                                                                                 hit                                                                   
lts__t_sectors_srcunit_l1_op_atom_evict_first_lookup_miss                        Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-first that 
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_l1_op_atom_evict_last                                     Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-last       
lts__t_sectors_srcunit_l1_op_atom_evict_last_lookup_hit                          Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-last that  
                                                                                                                 hit                                                                   
lts__t_sectors_srcunit_l1_op_atom_evict_last_lookup_miss                         Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-last that  
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_l1_op_atom_evict_normal                                   Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-normal     
                                                                                                                 (LRU)                                                                 
lts__t_sectors_srcunit_l1_op_atom_evict_normal_lookup_hit                        Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-normal     
                                                                                                                 (LRU) that hit                                                        
lts__t_sectors_srcunit_l1_op_atom_evict_normal_lookup_miss                       Counter         sector          # of LTS sectors from unit L1 for all atomics marked evict-normal     
                                                                                                                 (LRU) that missed                                                     
lts__t_sectors_srcunit_l1_op_atom_lookup_hit                                     Counter         sector          # of LTS sectors from unit L1 for all atomics that hit                
lts__t_sectors_srcunit_l1_op_atom_lookup_miss                                    Counter         sector          # of LTS sectors from unit L1 for all atomics that missed             
lts__t_sectors_srcunit_l1_op_membar                                              Counter         sector          # of LTS sectors from unit L1 for memory barriers                     
lts__t_sectors_srcunit_l1_op_membar_evict_first                                  Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-first  
lts__t_sectors_srcunit_l1_op_membar_evict_first_lookup_hit                       Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-first  
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_l1_op_membar_evict_first_lookup_miss                      Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-first  
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_l1_op_membar_evict_last                                   Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-last   
lts__t_sectors_srcunit_l1_op_membar_evict_last_lookup_hit                        Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-last   
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_l1_op_membar_evict_last_lookup_miss                       Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-last   
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_l1_op_membar_evict_normal                                 Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-normal 
                                                                                                                 (LRU)                                                                 
lts__t_sectors_srcunit_l1_op_membar_evict_normal_demote                          Counter         sector          # of LTS sectors from unit L1 for memory barriers marked              
                                                                                                                 evict-normal-demote                                                   
lts__t_sectors_srcunit_l1_op_membar_evict_normal_demote_lookup_hit               Counter         sector          # of LTS sectors from unit L1 for memory barriers marked              
                                                                                                                 evict-normal-demote that hit                                          
lts__t_sectors_srcunit_l1_op_membar_evict_normal_demote_lookup_miss              Counter         sector          # of LTS sectors from unit L1 for memory barriers marked              
                                                                                                                 evict-normal-demote that missed                                       
lts__t_sectors_srcunit_l1_op_membar_evict_normal_lookup_hit                      Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-normal 
                                                                                                                 (LRU) that hit                                                        
lts__t_sectors_srcunit_l1_op_membar_evict_normal_lookup_miss                     Counter         sector          # of LTS sectors from unit L1 for memory barriers marked evict-normal 
                                                                                                                 (LRU) that missed                                                     
lts__t_sectors_srcunit_l1_op_membar_lookup_hit                                   Counter         sector          # of LTS sectors from unit L1 for memory barriers that hit            
lts__t_sectors_srcunit_l1_op_membar_lookup_miss                                  Counter         sector          # of LTS sectors from unit L1 for memory barriers that missed         
lts__t_sectors_srcunit_l1_op_read                                                Counter         sector          # of LTS sectors from unit L1 for reads                               
lts__t_sectors_srcunit_l1_op_read_evict_first                                    Counter         sector          # of LTS sectors from unit L1 for reads marked evict-first            
lts__t_sectors_srcunit_l1_op_read_evict_first_lookup_hit                         Counter         sector          # of LTS sectors from unit L1 for reads marked evict-first that hit   
lts__t_sectors_srcunit_l1_op_read_evict_first_lookup_miss                        Counter         sector          # of LTS sectors from unit L1 for reads marked evict-first that missed
lts__t_sectors_srcunit_l1_op_read_evict_last                                     Counter         sector          # of LTS sectors from unit L1 for reads marked evict-last             
lts__t_sectors_srcunit_l1_op_read_evict_last_lookup_hit                          Counter         sector          # of LTS sectors from unit L1 for reads marked evict-last that hit    
lts__t_sectors_srcunit_l1_op_read_evict_last_lookup_miss                         Counter         sector          # of LTS sectors from unit L1 for reads marked evict-last that missed 
lts__t_sectors_srcunit_l1_op_read_evict_normal                                   Counter         sector          # of LTS sectors from unit L1 for reads marked evict-normal (LRU)     
lts__t_sectors_srcunit_l1_op_read_evict_normal_demote                            Counter         sector          # of LTS sectors from unit L1 for reads marked evict-normal-demote    
lts__t_sectors_srcunit_l1_op_read_evict_normal_demote_lookup_hit                 Counter         sector          # of LTS sectors from unit L1 for reads marked evict-normal-demote    
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_l1_op_read_evict_normal_demote_lookup_miss                Counter         sector          # of LTS sectors from unit L1 for reads marked evict-normal-demote    
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_l1_op_read_evict_normal_lookup_hit                        Counter         sector          # of LTS sectors from unit L1 for reads marked evict-normal (LRU)     
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_l1_op_read_evict_normal_lookup_miss                       Counter         sector          # of LTS sectors from unit L1 for reads marked evict-normal (LRU)     
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_l1_op_read_lookup_hit                                     Counter         sector          # of LTS sectors from unit L1 for reads that hit                      
lts__t_sectors_srcunit_l1_op_read_lookup_miss                                    Counter         sector          # of LTS sectors from unit L1 for reads that missed                   
lts__t_sectors_srcunit_l1_op_red                                                 Counter         sector          # of LTS sectors from unit L1 for reductions                          
lts__t_sectors_srcunit_l1_op_red_lookup_hit                                      Counter         sector          # of LTS sectors from unit L1 for reductions that hit                 
lts__t_sectors_srcunit_l1_op_red_lookup_miss                                     Counter         sector          # of LTS sectors from unit L1 for reductions that missed              
lts__t_sectors_srcunit_l1_op_write                                               Counter         sector          # of LTS sectors from unit L1 for writes                              
lts__t_sectors_srcunit_l1_op_write_evict_first                                   Counter         sector          # of LTS sectors from unit L1 for writes marked evict-first           
lts__t_sectors_srcunit_l1_op_write_evict_first_lookup_hit                        Counter         sector          # of LTS sectors from unit L1 for writes marked evict-first that hit  
lts__t_sectors_srcunit_l1_op_write_evict_first_lookup_miss                       Counter         sector          # of LTS sectors from unit L1 for writes marked evict-first that      
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_l1_op_write_evict_last                                    Counter         sector          # of LTS sectors from unit L1 for writes marked evict-last            
lts__t_sectors_srcunit_l1_op_write_evict_last_lookup_hit                         Counter         sector          # of LTS sectors from unit L1 for writes marked evict-last that hit   
lts__t_sectors_srcunit_l1_op_write_evict_last_lookup_miss                        Counter         sector          # of LTS sectors from unit L1 for writes marked evict-last that missed
lts__t_sectors_srcunit_l1_op_write_evict_normal                                  Counter         sector          # of LTS sectors from unit L1 for writes marked evict-normal (LRU)    
lts__t_sectors_srcunit_l1_op_write_evict_normal_demote                           Counter         sector          # of LTS sectors from unit L1 for writes marked evict-normal-demote   
lts__t_sectors_srcunit_l1_op_write_evict_normal_demote_lookup_hit                Counter         sector          # of LTS sectors from unit L1 for writes marked evict-normal-demote   
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_l1_op_write_evict_normal_demote_lookup_miss               Counter         sector          # of LTS sectors from unit L1 for writes marked evict-normal-demote   
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_l1_op_write_evict_normal_lookup_hit                       Counter         sector          # of LTS sectors from unit L1 for writes marked evict-normal (LRU)    
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_l1_op_write_evict_normal_lookup_miss                      Counter         sector          # of LTS sectors from unit L1 for writes marked evict-normal (LRU)    
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_l1_op_write_lookup_hit                                    Counter         sector          # of LTS sectors from unit L1 for writes that hit                     
lts__t_sectors_srcunit_l1_op_write_lookup_miss                                   Counter         sector          # of LTS sectors from unit L1 for writes that missed                  
lts__t_sectors_srcunit_ltcfabric                                                 Counter         sector          # of LTS sectors from LTC Fabric                                      
lts__t_sectors_srcunit_ltcfabric_aperture_device                                 Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_first                     Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-first                                                    
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_first_lookup_hit          Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-first that hit                                           
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_first_lookup_miss         Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-first that missed                                        
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_last                      Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-last                                                     
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_last_lookup_hit           Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-last that hit                                            
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_last_lookup_miss          Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-last that missed                                         
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_normal                    Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-normal (LRU)                                             
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_normal_demote             Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-normal-demote                                            
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_normal_demote_lookup_hit  Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-normal-demote that hit                                   
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_normal_demote_lookup_miss Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-normal-demote that missed                                
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_normal_lookup_hit         Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcunit_ltcfabric_aperture_device_evict_normal_lookup_miss        Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcunit_ltcfabric_aperture_device_lookup_hit                      Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_ltcfabric_aperture_device_lookup_miss                     Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem)     
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_membar                       Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 memory barriers                                                       
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_membar_lookup_hit            Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 memory barriers that hit                                              
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_membar_lookup_miss           Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 memory barriers that missed                                           
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_read                         Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 reads                                                                 
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_read_lookup_hit              Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 reads that hit                                                        
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_read_lookup_miss             Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 reads that missed                                                     
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_write                        Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 writes                                                                
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_write_lookup_hit             Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 writes that hit                                                       
lts__t_sectors_srcunit_ltcfabric_aperture_device_op_write_lookup_miss            Counter         sector          # of LTS sectors from LTC Fabric accessing device memory (vidmem) for 
                                                                                                                 writes that missed                                                    
lts__t_sectors_srcunit_ltcfabric_aperture_peer                                   Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_first                       Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-first                                                    
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_first_lookup_hit            Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-first that hit                                           
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_first_lookup_miss           Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-first that missed                                        
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_last                        Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-last                                                     
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_last_lookup_hit             Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-last that hit                                            
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_last_lookup_miss            Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-last that missed                                         
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_normal                      Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-normal (LRU)                                             
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_normal_demote               Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-normal-demote                                            
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_normal_demote_lookup_hit    Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-normal-demote that hit                                   
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_normal_demote_lookup_miss   Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-normal-demote that missed                                
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_normal_lookup_hit           Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcunit_ltcfabric_aperture_peer_evict_normal_lookup_miss          Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem)      
                                                                                                                 marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcunit_ltcfabric_aperture_peer_lookup_hit                        Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) that 
                                                                                                                 hit                                                                   
lts__t_sectors_srcunit_ltcfabric_aperture_peer_lookup_miss                       Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) that 
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_membar                         Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 memory barriers                                                       
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_membar_lookup_hit              Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 memory barriers that hit                                              
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_membar_lookup_miss             Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 memory barriers that missed                                           
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_read                           Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 reads                                                                 
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_read_lookup_hit                Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 reads that hit                                                        
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_read_lookup_miss               Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 reads that missed                                                     
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_write                          Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 writes                                                                
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_write_lookup_hit               Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 writes that hit                                                       
lts__t_sectors_srcunit_ltcfabric_aperture_peer_op_write_lookup_miss              Counter         sector          # of LTS sectors from LTC Fabric accessing peer memory (peermem) for  
                                                                                                                 writes that missed                                                    
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem                                 Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_first                     Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-first                                                    
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_first_lookup_hit          Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-first that hit                                           
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_first_lookup_miss         Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-first that missed                                        
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_last                      Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-last                                                     
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_last_lookup_hit           Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-last that hit                                            
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_last_lookup_miss          Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-last that missed                                         
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_normal                    Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-normal (LRU)                                             
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_normal_demote             Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-normal-demote                                            
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_normal_demote_lookup_hit  Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-normal-demote that hit                                   
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_normal_demote_lookup_miss Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-normal-demote that missed                                
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_normal_lookup_hit         Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_evict_normal_lookup_miss        Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_lookup_hit                      Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_lookup_miss                     Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem)     
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_membar                       Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 memory barriers                                                       
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_membar_lookup_hit            Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 memory barriers that hit                                              
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_membar_lookup_miss           Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 memory barriers that missed                                           
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_read                         Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 reads                                                                 
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_read_lookup_hit              Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 reads that hit                                                        
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_read_lookup_miss             Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 reads that missed                                                     
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_write                        Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 writes                                                                
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_write_lookup_hit             Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 writes that hit                                                       
lts__t_sectors_srcunit_ltcfabric_aperture_sysmem_op_write_lookup_miss            Counter         sector          # of LTS sectors from LTC Fabric accessing system memory (sysmem) for 
                                                                                                                 writes that missed                                                    
lts__t_sectors_srcunit_ltcfabric_evict_first                                     Counter         sector          # of LTS sectors from LTC Fabric marked evict-first                   
lts__t_sectors_srcunit_ltcfabric_evict_first_lookup_hit                          Counter         sector          # of LTS sectors from LTC Fabric marked evict-first that hit          
lts__t_sectors_srcunit_ltcfabric_evict_first_lookup_miss                         Counter         sector          # of LTS sectors from LTC Fabric marked evict-first that missed       
lts__t_sectors_srcunit_ltcfabric_evict_last                                      Counter         sector          # of LTS sectors from LTC Fabric marked evict-last                    
lts__t_sectors_srcunit_ltcfabric_evict_last_lookup_hit                           Counter         sector          # of LTS sectors from LTC Fabric marked evict-last that hit           
lts__t_sectors_srcunit_ltcfabric_evict_last_lookup_miss                          Counter         sector          # of LTS sectors from LTC Fabric marked evict-last that missed        
lts__t_sectors_srcunit_ltcfabric_evict_normal                                    Counter         sector          # of LTS sectors from LTC Fabric marked evict-normal (LRU)            
lts__t_sectors_srcunit_ltcfabric_evict_normal_demote                             Counter         sector          # of LTS sectors from LTC Fabric marked evict-normal-demote           
lts__t_sectors_srcunit_ltcfabric_evict_normal_demote_lookup_hit                  Counter         sector          # of LTS sectors from LTC Fabric marked evict-normal-demote that hit  
lts__t_sectors_srcunit_ltcfabric_evict_normal_demote_lookup_miss                 Counter         sector          # of LTS sectors from LTC Fabric marked evict-normal-demote that      
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_ltcfabric_evict_normal_lookup_hit                         Counter         sector          # of LTS sectors from LTC Fabric marked evict-normal (LRU) that hit   
lts__t_sectors_srcunit_ltcfabric_evict_normal_lookup_miss                        Counter         sector          # of LTS sectors from LTC Fabric marked evict-normal (LRU) that missed
lts__t_sectors_srcunit_ltcfabric_lookup_hit                                      Counter         sector          # of LTS sectors from LTC Fabric that hit                             
lts__t_sectors_srcunit_ltcfabric_lookup_miss                                     Counter         sector          # of LTS sectors from LTC Fabric that missed                          
lts__t_sectors_srcunit_ltcfabric_op_membar                                       Counter         sector          # of LTS sectors from LTC Fabric for memory barriers                  
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_first                           Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-first                                                           
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_first_lookup_hit                Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-first that hit                                                  
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_first_lookup_miss               Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-first that missed                                               
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_last                            Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked evict-last
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_last_lookup_hit                 Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-last that hit                                                   
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_last_lookup_miss                Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-last that missed                                                
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_normal                          Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-normal (LRU)                                                    
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_normal_demote                   Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-normal-demote                                                   
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_normal_demote_lookup_hit        Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-normal-demote that hit                                          
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_normal_demote_lookup_miss       Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-normal-demote that missed                                       
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_normal_lookup_hit               Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-normal (LRU) that hit                                           
lts__t_sectors_srcunit_ltcfabric_op_membar_evict_normal_lookup_miss              Counter         sector          # of LTS sectors from LTC Fabric for memory barriers marked           
                                                                                                                 evict-normal (LRU) that missed                                        
lts__t_sectors_srcunit_ltcfabric_op_membar_lookup_hit                            Counter         sector          # of LTS sectors from LTC Fabric for memory barriers that hit         
lts__t_sectors_srcunit_ltcfabric_op_membar_lookup_miss                           Counter         sector          # of LTS sectors from LTC Fabric for memory barriers that missed      
lts__t_sectors_srcunit_ltcfabric_op_read                                         Counter         sector          # of LTS sectors from LTC Fabric for reads                            
lts__t_sectors_srcunit_ltcfabric_op_read_evict_first                             Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-first         
lts__t_sectors_srcunit_ltcfabric_op_read_evict_first_lookup_hit                  Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-first that hit
lts__t_sectors_srcunit_ltcfabric_op_read_evict_first_lookup_miss                 Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-first that    
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_ltcfabric_op_read_evict_last                              Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-last          
lts__t_sectors_srcunit_ltcfabric_op_read_evict_last_lookup_hit                   Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-last that hit 
lts__t_sectors_srcunit_ltcfabric_op_read_evict_last_lookup_miss                  Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-last that     
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_ltcfabric_op_read_evict_normal                            Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-normal (LRU)  
lts__t_sectors_srcunit_ltcfabric_op_read_evict_normal_demote                     Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-normal-demote 
lts__t_sectors_srcunit_ltcfabric_op_read_evict_normal_demote_lookup_hit          Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-normal-demote 
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_ltcfabric_op_read_evict_normal_demote_lookup_miss         Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-normal-demote 
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_ltcfabric_op_read_evict_normal_lookup_hit                 Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-normal (LRU)  
                                                                                                                 that hit                                                              
lts__t_sectors_srcunit_ltcfabric_op_read_evict_normal_lookup_miss                Counter         sector          # of LTS sectors from LTC Fabric for reads marked evict-normal (LRU)  
                                                                                                                 that missed                                                           
lts__t_sectors_srcunit_ltcfabric_op_read_lookup_hit                              Counter         sector          # of LTS sectors from LTC Fabric for reads that hit                   
lts__t_sectors_srcunit_ltcfabric_op_read_lookup_miss                             Counter         sector          # of LTS sectors from LTC Fabric for reads that missed                
lts__t_sectors_srcunit_ltcfabric_op_write                                        Counter         sector          # of LTS sectors from LTC Fabric for writes                           
lts__t_sectors_srcunit_ltcfabric_op_write_evict_first                            Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-first        
lts__t_sectors_srcunit_ltcfabric_op_write_evict_first_lookup_hit                 Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-first that   
                                                                                                                 hit                                                                   
lts__t_sectors_srcunit_ltcfabric_op_write_evict_first_lookup_miss                Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-first that   
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_ltcfabric_op_write_evict_last                             Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-last         
lts__t_sectors_srcunit_ltcfabric_op_write_evict_last_lookup_hit                  Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-last that hit
lts__t_sectors_srcunit_ltcfabric_op_write_evict_last_lookup_miss                 Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-last that    
                                                                                                                 missed                                                                
lts__t_sectors_srcunit_ltcfabric_op_write_evict_normal                           Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-normal (LRU) 
lts__t_sectors_srcunit_ltcfabric_op_write_evict_normal_demote                    Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-normal-demote
lts__t_sectors_srcunit_ltcfabric_op_write_evict_normal_demote_lookup_hit         Counter         sector          # of LTS sectors from LTC Fabric for writes marked                    
                                                                                                                 evict-normal-demote that hit                                          
lts__t_sectors_srcunit_ltcfabric_op_write_evict_normal_demote_lookup_miss   Counter         sector          # of LTS sectors from LTC Fabric for writes marked                    
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcunit_ltcfabric_op_write_evict_normal_lookup_hit           Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-normal (LRU) 
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_ltcfabric_op_write_evict_normal_lookup_miss          Counter         sector          # of LTS sectors from LTC Fabric for writes marked evict-normal (LRU) 
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_ltcfabric_op_write_lookup_hit                        Counter         sector          # of LTS sectors from LTC Fabric for writes that hit                  
lts__t_sectors_srcunit_ltcfabric_op_write_lookup_miss                       Counter         sector          # of LTS sectors from LTC Fabric for writes that missed               
lts__t_sectors_srcunit_tex                                                  Counter         sector          # of LTS sectors from unit TEX                                        
lts__t_sectors_srcunit_tex_aperture_device                                  Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
lts__t_sectors_srcunit_tex_aperture_device_evict_first                      Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-first                                                    
lts__t_sectors_srcunit_tex_aperture_device_evict_first_lookup_hit           Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_srcunit_tex_aperture_device_evict_first_lookup_miss          Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_srcunit_tex_aperture_device_evict_last                       Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-last                                                     
lts__t_sectors_srcunit_tex_aperture_device_evict_last_lookup_hit            Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_srcunit_tex_aperture_device_evict_last_lookup_miss           Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_srcunit_tex_aperture_device_evict_normal                     Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_srcunit_tex_aperture_device_evict_normal_demote              Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_srcunit_tex_aperture_device_evict_normal_demote_lookup_hit   Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_srcunit_tex_aperture_device_evict_normal_demote_lookup_miss  Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_srcunit_tex_aperture_device_evict_normal_lookup_hit          Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcunit_tex_aperture_device_evict_normal_lookup_miss         Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcunit_tex_aperture_device_lookup_hit                       Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) that  
                                                                                                            hit                                                                   
lts__t_sectors_srcunit_tex_aperture_device_lookup_miss                      Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) that  
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_aperture_device_op_atom                          Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            all atomics                                                           
lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_alu                  Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_alu_lookup_hit       Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_alu_lookup_miss      Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_cas                  Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_cas_lookup_hit       Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcunit_tex_aperture_device_op_atom_dot_cas_lookup_miss      Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcunit_tex_aperture_device_op_atom_lookup_hit               Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcunit_tex_aperture_device_op_atom_lookup_miss              Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcunit_tex_aperture_device_op_membar                        Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            memory barriers                                                       
lts__t_sectors_srcunit_tex_aperture_device_op_membar_lookup_hit             Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcunit_tex_aperture_device_op_membar_lookup_miss            Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcunit_tex_aperture_device_op_read                          Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            reads                                                                 
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_hit               Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            reads that hit                                                        
lts__t_sectors_srcunit_tex_aperture_device_op_read_lookup_miss              Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            reads that missed                                                     
lts__t_sectors_srcunit_tex_aperture_device_op_red                           Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            reductions                                                            
lts__t_sectors_srcunit_tex_aperture_device_op_red_lookup_hit                Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcunit_tex_aperture_device_op_red_lookup_miss               Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            reductions that missed                                                
lts__t_sectors_srcunit_tex_aperture_device_op_write                         Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            writes                                                                
lts__t_sectors_srcunit_tex_aperture_device_op_write_lookup_hit              Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            writes that hit                                                       
lts__t_sectors_srcunit_tex_aperture_device_op_write_lookup_miss             Counter         sector          # of LTS sectors from unit TEX accessing device memory (vidmem) for   
                                                                                                            writes that missed                                                    
lts__t_sectors_srcunit_tex_aperture_peer                                    Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem)        
lts__t_sectors_srcunit_tex_aperture_peer_evict_first                        Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-first                                                           
lts__t_sectors_srcunit_tex_aperture_peer_evict_first_lookup_hit             Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-first that hit                                                  
lts__t_sectors_srcunit_tex_aperture_peer_evict_first_lookup_miss            Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-first that missed                                               
lts__t_sectors_srcunit_tex_aperture_peer_evict_last                         Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-last                                                            
lts__t_sectors_srcunit_tex_aperture_peer_evict_last_lookup_hit              Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-last that hit                                                   
lts__t_sectors_srcunit_tex_aperture_peer_evict_last_lookup_miss             Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-last that missed                                                
lts__t_sectors_srcunit_tex_aperture_peer_evict_normal                       Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcunit_tex_aperture_peer_evict_normal_demote                Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcunit_tex_aperture_peer_evict_normal_demote_lookup_hit     Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcunit_tex_aperture_peer_evict_normal_demote_lookup_miss    Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcunit_tex_aperture_peer_evict_normal_lookup_hit            Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcunit_tex_aperture_peer_evict_normal_lookup_miss           Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) marked 
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcunit_tex_aperture_peer_lookup_hit                         Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) that   
                                                                                                            hit                                                                   
lts__t_sectors_srcunit_tex_aperture_peer_lookup_miss                        Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) that   
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_aperture_peer_op_atom                            Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            all atomics                                                           
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_dot_alu                    Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_dot_alu_lookup_hit         Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_dot_alu_lookup_miss        Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_dot_cas                    Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_dot_cas_lookup_hit         Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_dot_cas_lookup_miss        Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_lookup_hit                 Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcunit_tex_aperture_peer_op_atom_lookup_miss                Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcunit_tex_aperture_peer_op_membar                          Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            memory barriers                                                       
lts__t_sectors_srcunit_tex_aperture_peer_op_membar_lookup_hit               Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcunit_tex_aperture_peer_op_membar_lookup_miss              Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcunit_tex_aperture_peer_op_read                            Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            reads                                                                 
lts__t_sectors_srcunit_tex_aperture_peer_op_read_lookup_hit                 Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            reads that hit                                                        
lts__t_sectors_srcunit_tex_aperture_peer_op_read_lookup_miss                Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            reads that missed                                                     
lts__t_sectors_srcunit_tex_aperture_peer_op_red                             Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            reductions                                                            
lts__t_sectors_srcunit_tex_aperture_peer_op_red_lookup_hit                  Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcunit_tex_aperture_peer_op_red_lookup_miss                 Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            reductions that missed                                                
lts__t_sectors_srcunit_tex_aperture_peer_op_write                           Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            writes                                                                
lts__t_sectors_srcunit_tex_aperture_peer_op_write_lookup_hit                Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            writes that hit                                                       
lts__t_sectors_srcunit_tex_aperture_peer_op_write_lookup_miss               Counter         sector          # of LTS sectors from unit TEX accessing peer memory (peermem) for    
                                                                                                            writes that missed                                                    
lts__t_sectors_srcunit_tex_aperture_sysmem                                  Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_first                      Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-first                                                    
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_first_lookup_hit           Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-first that hit                                           
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_first_lookup_miss          Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-first that missed                                        
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_last                       Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-last                                                     
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_last_lookup_hit            Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-last that hit                                            
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_last_lookup_miss           Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-last that missed                                         
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_normal                     Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU)                                             
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_normal_demote              Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote                                            
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_normal_demote_lookup_hit   Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote that hit                                   
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_normal_demote_lookup_miss  Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-normal-demote that missed                                
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_normal_lookup_hit          Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU) that hit                                    
lts__t_sectors_srcunit_tex_aperture_sysmem_evict_normal_lookup_miss         Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem)       
                                                                                                            marked evict-normal (LRU) that missed                                 
lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_hit                       Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) that  
                                                                                                            hit                                                                   
lts__t_sectors_srcunit_tex_aperture_sysmem_lookup_miss                      Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) that  
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom                          Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            all atomics                                                           
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_dot_alu                  Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS)                                                  
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_dot_alu_lookup_hit       Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS) that hit                                         
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_dot_alu_lookup_miss      Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            atomic ALU (non-CAS) that missed                                      
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_dot_cas                  Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            atomic CAS                                                            
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_dot_cas_lookup_hit       Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            atomic CAS that hit                                                   
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_dot_cas_lookup_miss      Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            atomic CAS that missed                                                
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_lookup_hit               Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            all atomics that hit                                                  
lts__t_sectors_srcunit_tex_aperture_sysmem_op_atom_lookup_miss              Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            all atomics that missed                                               
lts__t_sectors_srcunit_tex_aperture_sysmem_op_membar                        Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            memory barriers                                                       
lts__t_sectors_srcunit_tex_aperture_sysmem_op_membar_lookup_hit             Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            memory barriers that hit                                              
lts__t_sectors_srcunit_tex_aperture_sysmem_op_membar_lookup_miss            Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            memory barriers that missed                                           
lts__t_sectors_srcunit_tex_aperture_sysmem_op_read                          Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            reads                                                                 
lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_hit               Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            reads that hit                                                        
lts__t_sectors_srcunit_tex_aperture_sysmem_op_read_lookup_miss              Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            reads that missed                                                     
lts__t_sectors_srcunit_tex_aperture_sysmem_op_red                           Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            reductions                                                            
lts__t_sectors_srcunit_tex_aperture_sysmem_op_red_lookup_hit                Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            reductions that hit                                                   
lts__t_sectors_srcunit_tex_aperture_sysmem_op_red_lookup_miss               Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            reductions that missed                                                
lts__t_sectors_srcunit_tex_aperture_sysmem_op_write                         Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            writes                                                                
lts__t_sectors_srcunit_tex_aperture_sysmem_op_write_lookup_hit              Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            writes that hit                                                       
lts__t_sectors_srcunit_tex_aperture_sysmem_op_write_lookup_miss             Counter         sector          # of LTS sectors from unit TEX accessing system memory (sysmem) for   
                                                                                                            writes that missed                                                    
lts__t_sectors_srcunit_tex_evict_first                                      Counter         sector          # of LTS sectors from unit TEX marked evict-first                     
lts__t_sectors_srcunit_tex_evict_first_lookup_hit                           Counter         sector          # of LTS sectors from unit TEX marked evict-first that hit            
lts__t_sectors_srcunit_tex_evict_first_lookup_miss                          Counter         sector          # of LTS sectors from unit TEX marked evict-first that missed         
lts__t_sectors_srcunit_tex_evict_last                                       Counter         sector          # of LTS sectors from unit TEX marked evict-last                      
lts__t_sectors_srcunit_tex_evict_last_lookup_hit                            Counter         sector          # of LTS sectors from unit TEX marked evict-last that hit             
lts__t_sectors_srcunit_tex_evict_last_lookup_miss                           Counter         sector          # of LTS sectors from unit TEX marked evict-last that missed          
lts__t_sectors_srcunit_tex_evict_normal                                     Counter         sector          # of LTS sectors from unit TEX marked evict-normal (LRU)              
lts__t_sectors_srcunit_tex_evict_normal_demote                              Counter         sector          # of LTS sectors from unit TEX marked evict-normal-demote             
lts__t_sectors_srcunit_tex_evict_normal_demote_lookup_hit                   Counter         sector          # of LTS sectors from unit TEX marked evict-normal-demote that hit    
lts__t_sectors_srcunit_tex_evict_normal_demote_lookup_miss                  Counter         sector          # of LTS sectors from unit TEX marked evict-normal-demote that missed 
lts__t_sectors_srcunit_tex_evict_normal_lookup_hit                          Counter         sector          # of LTS sectors from unit TEX marked evict-normal (LRU) that hit     
lts__t_sectors_srcunit_tex_evict_normal_lookup_miss                         Counter         sector          # of LTS sectors from unit TEX marked evict-normal (LRU) that missed  
lts__t_sectors_srcunit_tex_lookup_hit                                       Counter         sector          # of LTS sectors from unit TEX that hit                               
lts__t_sectors_srcunit_tex_lookup_miss                                      Counter         sector          # of LTS sectors from unit TEX that missed                            
lts__t_sectors_srcunit_tex_op_atom                                          Counter         sector          # of LTS sectors from unit TEX for all atomics                        
lts__t_sectors_srcunit_tex_op_atom_dot_alu                                  Counter         sector          # of LTS sectors from unit TEX for atomic ALU (non-CAS)               
lts__t_sectors_srcunit_tex_op_atom_dot_alu_lookup_hit                       Counter         sector          # of LTS sectors from unit TEX for atomic ALU (non-CAS) that hit      
lts__t_sectors_srcunit_tex_op_atom_dot_alu_lookup_miss                      Counter         sector          # of LTS sectors from unit TEX for atomic ALU (non-CAS) that missed   
lts__t_sectors_srcunit_tex_op_atom_dot_cas                                  Counter         sector          # of LTS sectors from unit TEX for atomic CAS                         
lts__t_sectors_srcunit_tex_op_atom_dot_cas_lookup_hit                       Counter         sector          # of LTS sectors from unit TEX for atomic CAS that hit                
lts__t_sectors_srcunit_tex_op_atom_dot_cas_lookup_miss                      Counter         sector          # of LTS sectors from unit TEX for atomic CAS that missed             
lts__t_sectors_srcunit_tex_op_atom_evict_first                              Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-first     
lts__t_sectors_srcunit_tex_op_atom_evict_first_lookup_hit                   Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-first     
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_atom_evict_first_lookup_miss                  Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-first     
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_atom_evict_last                               Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-last      
lts__t_sectors_srcunit_tex_op_atom_evict_last_lookup_hit                    Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-last that 
                                                                                                            hit                                                                   
lts__t_sectors_srcunit_tex_op_atom_evict_last_lookup_miss                   Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-last that 
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_op_atom_evict_normal                             Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-normal    
                                                                                                            (LRU)                                                                 
lts__t_sectors_srcunit_tex_op_atom_evict_normal_lookup_hit                  Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-normal    
                                                                                                            (LRU) that hit                                                        
lts__t_sectors_srcunit_tex_op_atom_evict_normal_lookup_miss                 Counter         sector          # of LTS sectors from unit TEX for all atomics marked evict-normal    
                                                                                                            (LRU) that missed                                                     
lts__t_sectors_srcunit_tex_op_atom_lookup_hit                               Counter         sector          # of LTS sectors from unit TEX for all atomics that hit               
lts__t_sectors_srcunit_tex_op_atom_lookup_miss                              Counter         sector          # of LTS sectors from unit TEX for all atomics that missed            
lts__t_sectors_srcunit_tex_op_membar                                        Counter         sector          # of LTS sectors from unit TEX for memory barriers                    
lts__t_sectors_srcunit_tex_op_membar_evict_first                            Counter         sector          # of LTS sectors from unit TEX for memory barriers marked evict-first 
lts__t_sectors_srcunit_tex_op_membar_evict_first_lookup_hit                 Counter         sector          # of LTS sectors from unit TEX for memory barriers marked evict-first 
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_membar_evict_first_lookup_miss                Counter         sector          # of LTS sectors from unit TEX for memory barriers marked evict-first 
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_membar_evict_last                             Counter         sector          # of LTS sectors from unit TEX for memory barriers marked evict-last  
lts__t_sectors_srcunit_tex_op_membar_evict_last_lookup_hit                  Counter         sector          # of LTS sectors from unit TEX for memory barriers marked evict-last  
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_membar_evict_last_lookup_miss                 Counter         sector          # of LTS sectors from unit TEX for memory barriers marked evict-last  
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_membar_evict_normal                           Counter         sector          # of LTS sectors from unit TEX for memory barriers marked             
                                                                                                            evict-normal (LRU)                                                    
lts__t_sectors_srcunit_tex_op_membar_evict_normal_demote                    Counter         sector          # of LTS sectors from unit TEX for memory barriers marked             
                                                                                                            evict-normal-demote                                                   
lts__t_sectors_srcunit_tex_op_membar_evict_normal_demote_lookup_hit         Counter         sector          # of LTS sectors from unit TEX for memory barriers marked             
                                                                                                            evict-normal-demote that hit                                          
lts__t_sectors_srcunit_tex_op_membar_evict_normal_demote_lookup_miss        Counter         sector          # of LTS sectors from unit TEX for memory barriers marked             
                                                                                                            evict-normal-demote that missed                                       
lts__t_sectors_srcunit_tex_op_membar_evict_normal_lookup_hit                Counter         sector          # of LTS sectors from unit TEX for memory barriers marked             
                                                                                                            evict-normal (LRU) that hit                                           
lts__t_sectors_srcunit_tex_op_membar_evict_normal_lookup_miss               Counter         sector          # of LTS sectors from unit TEX for memory barriers marked             
                                                                                                            evict-normal (LRU) that missed                                        
lts__t_sectors_srcunit_tex_op_membar_lookup_hit                             Counter         sector          # of LTS sectors from unit TEX for memory barriers that hit           
lts__t_sectors_srcunit_tex_op_membar_lookup_miss                            Counter         sector          # of LTS sectors from unit TEX for memory barriers that missed        
lts__t_sectors_srcunit_tex_op_read                                          Counter         sector          # of LTS sectors from unit TEX for reads                              
lts__t_sectors_srcunit_tex_op_read_evict_first                              Counter         sector          # of LTS sectors from unit TEX for reads marked evict-first           
lts__t_sectors_srcunit_tex_op_read_evict_first_lookup_hit                   Counter         sector          # of LTS sectors from unit TEX for reads marked evict-first that hit  
lts__t_sectors_srcunit_tex_op_read_evict_first_lookup_miss                  Counter         sector          # of LTS sectors from unit TEX for reads marked evict-first that      
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_op_read_evict_last                               Counter         sector          # of LTS sectors from unit TEX for reads marked evict-last            
lts__t_sectors_srcunit_tex_op_read_evict_last_lookup_hit                    Counter         sector          # of LTS sectors from unit TEX for reads marked evict-last that hit   
lts__t_sectors_srcunit_tex_op_read_evict_last_lookup_miss                   Counter         sector          # of LTS sectors from unit TEX for reads marked evict-last that missed
lts__t_sectors_srcunit_tex_op_read_evict_normal                             Counter         sector          # of LTS sectors from unit TEX for reads marked evict-normal (LRU)    
lts__t_sectors_srcunit_tex_op_read_evict_normal_demote                      Counter         sector          # of LTS sectors from unit TEX for reads marked evict-normal-demote   
lts__t_sectors_srcunit_tex_op_read_evict_normal_demote_lookup_hit           Counter         sector          # of LTS sectors from unit TEX for reads marked evict-normal-demote   
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_read_evict_normal_demote_lookup_miss          Counter         sector          # of LTS sectors from unit TEX for reads marked evict-normal-demote   
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_read_evict_normal_lookup_hit                  Counter         sector          # of LTS sectors from unit TEX for reads marked evict-normal (LRU)    
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_read_evict_normal_lookup_miss                 Counter         sector          # of LTS sectors from unit TEX for reads marked evict-normal (LRU)    
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_read_lookup_hit                               Counter         sector          # of LTS sectors from unit TEX for reads that hit                     
lts__t_sectors_srcunit_tex_op_read_lookup_miss                              Counter         sector          # of LTS sectors from unit TEX for reads that missed                  
lts__t_sectors_srcunit_tex_op_red                                           Counter         sector          # of LTS sectors from unit TEX for reductions                         
lts__t_sectors_srcunit_tex_op_red_lookup_hit                                Counter         sector          # of LTS sectors from unit TEX for reductions that hit                
lts__t_sectors_srcunit_tex_op_red_lookup_miss                               Counter         sector          # of LTS sectors from unit TEX for reductions that missed             
lts__t_sectors_srcunit_tex_op_write                                         Counter         sector          # of LTS sectors from unit TEX for writes                             
lts__t_sectors_srcunit_tex_op_write_evict_first                             Counter         sector          # of LTS sectors from unit TEX for writes marked evict-first          
lts__t_sectors_srcunit_tex_op_write_evict_first_lookup_hit                  Counter         sector          # of LTS sectors from unit TEX for writes marked evict-first that hit 
lts__t_sectors_srcunit_tex_op_write_evict_first_lookup_miss                 Counter         sector          # of LTS sectors from unit TEX for writes marked evict-first that     
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_op_write_evict_last                              Counter         sector          # of LTS sectors from unit TEX for writes marked evict-last           
lts__t_sectors_srcunit_tex_op_write_evict_last_lookup_hit                   Counter         sector          # of LTS sectors from unit TEX for writes marked evict-last that hit  
lts__t_sectors_srcunit_tex_op_write_evict_last_lookup_miss                  Counter         sector          # of LTS sectors from unit TEX for writes marked evict-last that      
                                                                                                            missed                                                                
lts__t_sectors_srcunit_tex_op_write_evict_normal                            Counter         sector          # of LTS sectors from unit TEX for writes marked evict-normal (LRU)   
lts__t_sectors_srcunit_tex_op_write_evict_normal_demote                     Counter         sector          # of LTS sectors from unit TEX for writes marked evict-normal-demote  
lts__t_sectors_srcunit_tex_op_write_evict_normal_demote_lookup_hit          Counter         sector          # of LTS sectors from unit TEX for writes marked evict-normal-demote  
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_write_evict_normal_demote_lookup_miss         Counter         sector          # of LTS sectors from unit TEX for writes marked evict-normal-demote  
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_write_evict_normal_lookup_hit                 Counter         sector          # of LTS sectors from unit TEX for writes marked evict-normal (LRU)   
                                                                                                            that hit                                                              
lts__t_sectors_srcunit_tex_op_write_evict_normal_lookup_miss                Counter         sector          # of LTS sectors from unit TEX for writes marked evict-normal (LRU)   
                                                                                                            that missed                                                           
lts__t_sectors_srcunit_tex_op_write_lookup_hit                              Counter         sector          # of LTS sectors from unit TEX for writes that hit                    
lts__t_sectors_srcunit_tex_op_write_lookup_miss                             Counter         sector          # of LTS sectors from unit TEX for writes that missed                 
lts__t_tag_requests                                                         Counter                         # of tag requests sent to LTS                                         
lts__t_tag_requests_hit                                                     Counter                         # of LTS requests with tag hit                                        
lts__t_tag_requests_miss                                                    Counter                         # of LTS requests with tag miss                                       
lts__throughput                                                             Throughput      %               LTS throughput                                                        
lts__throughput_internal_activity                                           Throughput      %               LTS throughput, internal activity                                     
lts__xbar2lts_cycles_active                                                 Counter         cycle           # of cycles where interface from XBAR to LTS was active               
pcie__cycles_active                                                         Counter         cycle           alias of pcie__cycles_elapsed                                         
pcie__cycles_elapsed                                                        Counter         cycle           # of cycles elapsed on PCIe                                           
pcie__cycles_in_frame                                                       Counter         cycle           # of cycles in user-defined frame                                     
pcie__cycles_in_region                                                      Counter         cycle           # of cycles in user-defined region                                    
pcie__read_bytes                                                            Counter         byte            # of bytes of PCIe read data, at 512B granularity                     
pcie__write_bytes                                                           Counter         byte            # of bytes of PCIe write data, at 512B granularity                    
sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime         Ratio                           average # of active predicated-on threads per instruction executed    
sm__average_threads_launched_per_warp                                       Ratio           thread/warp     # of threads launched per warp                                        
sm__average_threads_launched_per_warp_shader_cs                             Ratio           thread/warp     average # of CS threads launched per CS warp                          
sm__ctas_active                                                             Counter         block           cumulative # of CTAs in flight                                        
sm__ctas_launched                                                           Counter         block           # of CTAs launched                                                    
sm__ctas_launched_total                                                     Counter         block           # of CTAs launched including preemption-restore events                
sm__ctas_restored                                                           Counter         block           # of CTA preemption-restore events                                    
sm__cycles_active                                                           Counter         cycle           # of cycles with at least one warp in flight                          
sm__cycles_active_shader_cs                                                 Counter         cycle           # of cycles where CS warps were resident                              
sm__cycles_elapsed                                                          Counter         cycle           # of cycles elapsed on SM                                             
sm__cycles_in_frame                                                         Counter         cycle           # of cycles in user-defined frame                                     
sm__cycles_in_region                                                        Counter         cycle           # of cycles in user-defined region                                    
sm__idc_divergent_instruction_replays                                       Counter         inst            # of IDC instruction replays due to address divergence                
sm__idc_divergent_instructions                                              Counter         inst            # of instructions sent to IDC that had address divergence             
sm__inst_executed                                                           Counter         inst            # of warp instructions executed                                       
sm__inst_executed_op_ldsm                                                   Counter         inst            # of warp instructions executed: LDSM                                 
sm__inst_executed_pipe_adu                                                  Counter         inst            # of warp instructions executed by adu pipe                           
sm__inst_executed_pipe_alu                                                  Counter         inst            # of warp instructions executed by alu pipe                           
sm__inst_executed_pipe_cbu                                                  Counter         inst            # of warp instructions executed by cbu pipe                           
sm__inst_executed_pipe_cbu_pred_off_all                                     Counter         inst            # of warp instructions executed by cbu pipe with all threads          
                                                                                                            predicated off                                                        
sm__inst_executed_pipe_cbu_pred_on_any                                      Counter         inst            # of warp instructions executed by cbu pipe with at least 1 thread    
                                                                                                            predicated on                                                         
sm__inst_executed_pipe_fma                                                  Counter         inst            # of warp instructions executed by fma pipe                           
sm__inst_executed_pipe_fp16                                                 Counter         inst            # of warp instructions executed by fp16 pipe                          
sm__inst_executed_pipe_fp64                                                 Counter         inst            # of warp instructions executed by fp64 pipe                          
sm__inst_executed_pipe_ipa                                                  Counter         inst            # of warp instructions executed by ipa pipe                           
sm__inst_executed_pipe_lsu                                                  Counter         inst            # of warp instructions executed by lsu pipe                           
sm__inst_executed_pipe_tensor                                               Counter         inst            # of warp instructions executed by tensor pipe                        
sm__inst_executed_pipe_tensor_op_dmma                                       Counter         inst            # of warp instructions executed by tensor pipe (DMMA ops)             
sm__inst_executed_pipe_tensor_op_hmma                                       Counter         inst            # of warp instructions executed by tensor pipe (HMMA ops)             
sm__inst_executed_pipe_tensor_op_hmma_type_hfma2                            Counter         inst            # of warp instructions executed by tensor pipe (HFMA2.MMA ops)        
sm__inst_executed_pipe_tensor_op_imma                                       Counter         inst            # of warp instructions executed by tensor pipe (IMMA ops)             
sm__inst_executed_pipe_tex                                                  Counter         inst            # of warp instructions executed by tex pipe                           
sm__inst_executed_pipe_uniform                                              Counter         inst            # of warp instructions executed by uniform pipe                       
sm__inst_executed_pipe_xu                                                   Counter         inst            # of warp instructions executed by xu pipe                            
sm__inst_issued                                                             Counter         inst            # of warp instructions issued                                         
sm__instruction_throughput                                                  Throughput      %               SM core instruction throughput assuming ideal load balancing across   
                                                                                                            SMSPs                                                                 
sm__instruction_throughput_internal_activity                                Throughput      %               SM core instruction throughput assuming ideal load balancing across   
                                                                                                            SMSPs, internal activity                                              
sm__issue_active                                                            Counter         cycle           # of cycles where an SMSP issued an instruction                       
sm__memory_throughput                                                       Throughput      %               SM memory instruction throughput assuming ideal load balancing across 
                                                                                                            SMSPs                                                                 
sm__memory_throughput_internal_activity                                     Throughput      %               SM memory instruction throughput assuming ideal load balancing across 
                                                                                                            SMSPs, internal activity                                              
sm__mio2rf_writeback_active                                                 Counter         cycle           # of cycles where the MIO to register file writeback interface was    
                                                                                                            active                                                                
sm__mio_inst_issued                                                         Counter         inst            # of instructions issued from MIOC to MIO                             
sm__mio_pq_read_cycles_active                                               Counter         cycle           # of cycles where MIOP PQ sent register operands to a pipeline        
sm__mio_pq_read_cycles_active_pipe_adu                                      Counter         cycle           # of cycles where MIOP PQ sent register operands to the adu pipe      
sm__mio_pq_read_cycles_active_pipe_lsu                                      Counter         cycle           # of cycles where MIOP PQ sent register operands to the lsu pipe      
sm__mio_pq_read_cycles_active_pipe_lsu_op_pixout                            Counter         cycle           # of cycles where MIOP PQ sent register operands to pixout            
sm__mio_pq_read_cycles_active_pipe_tex                                      Counter         cycle           # of cycles where MIOP PQ sent register operands to the tex pipe      
sm__mio_pq_write_cycles_active                                              Counter         cycle           # of cycles where register operands from the register file were       
                                                                                                            written to MIO PQ                                                     
sm__mio_pq_write_cycles_active_pipe_lsu                                     Counter         cycle           # of cycles where register operands from the register file were       
                                                                                                            written to MIO PQ, for the lsu pipe                                   
sm__mio_pq_write_cycles_active_pipe_tex                                     Counter         cycle           # of cycles where register operands from the register file were       
                                                                                                            written to MIO PQ, for the tex pipe                                   
sm__mioc_inst_issued                                                        Counter         inst            # of instructions issued from SMSP to MIO Control stage               
sm__pipe_alu_cycles_active                                                  Counter         cycle           # of cycles where alu pipe was active                                 
sm__pipe_fma_cycles_active                                                  Counter         cycle           # of cycles where fma pipe was active                                 
sm__pipe_fp64_cycles_active                                                 Counter         cycle           # of cycles where fp64 pipe was active                                
sm__pipe_shared_cycles_active                                               Counter         cycle           # of cycles where the shared pipe (fp64+tensor) was active            
sm__pipe_shared_cycles_active_realtime                                      Counter         cycle           # of cycles where the shared pipe (fp64+tensor) was active            
sm__pipe_tensor_cycles_active                                               Counter         cycle           # of cycles where tensor pipe was active                              
sm__pipe_tensor_op_dmma_cycles_active                                       Counter         inst            # of cycles where tensor pipe was active (DMMA ops)                   
sm__pipe_tensor_op_hmma_cycles_active                                       Counter         cycle           # of cycles where tensor pipe was active (HMMA ops)                   
sm__pipe_tensor_op_imma_cycles_active                                       Counter         cycle           # of cycles where tensor pipe was active (IMMA ops)                   
sm__sass_average_branch_targets_threads_uniform                             Ratio                           proportion of branch targets where all active threads selected the    
                                                                                                            same branch target                                                    
sm__sass_branch_targets                                                     Counter                         # of unique branch targets assigned to the PC                         
sm__sass_branch_targets_threads_divergent                                   Counter                         incremented only when there are two or more active threads with       
                                                                                                            different branch target                                               
sm__sass_branch_targets_threads_uniform                                     Counter                         # of branch executions where all active threads selected the same     
                                                                                                            branch target                                                         
sm__sass_data_bytes_mem_global                                              Counter         byte            # of bytes required for global operations                             
sm__sass_data_bytes_mem_global_op_atom                                      Counter         byte            # of bytes required for global atom                                   
sm__sass_data_bytes_mem_global_op_ld                                        Counter         byte            # of bytes required for global loads                                  
sm__sass_data_bytes_mem_global_op_ldgsts                                    Counter         byte            # of bytes required for LDGSTS global loads                           
sm__sass_data_bytes_mem_global_op_ldgsts_cache_access                       Counter         byte            # of bytes required for LDGSTS.ACCESS global loads                    
sm__sass_data_bytes_mem_global_op_ldgsts_cache_bypass                       Counter         byte            # of bytes required for LDGSTS.BYPASS global loads                    
sm__sass_data_bytes_mem_global_op_red                                       Counter         byte            # of bytes required for global reductions                             
sm__sass_data_bytes_mem_global_op_st                                        Counter         byte            # of bytes required for global stores                                 
sm__sass_data_bytes_mem_local                                               Counter         byte            # of bytes required for local operations                              
sm__sass_data_bytes_mem_local_op_ld                                         Counter         byte            # of bytes required for local loads                                   
sm__sass_data_bytes_mem_local_op_st                                         Counter         byte            # of bytes required for local stores                                  
sm__sass_data_bytes_mem_shared                                              Counter         byte            # of shared memory bytes required for LDS, LD, STS, ST, ATOMS, ATOM,  
                                                                                                            LDSM, LDGSTS                                                          
sm__sass_data_bytes_mem_shared_op_atom                                      Counter         byte            # of shared memory bytes required for ATOMS, ATOM                     
sm__sass_data_bytes_mem_shared_op_ld                                        Counter         byte            # of shared memory bytes required for LDS, LD                         
sm__sass_data_bytes_mem_shared_op_ldgsts                                    Counter         byte            # of shared memory bytes required for LDGSTS                          
sm__sass_data_bytes_mem_shared_op_ldgsts_cache_access                       Counter         byte            # of shared memory bytes required for LDGSTS.ACCESS                   
sm__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass                       Counter         byte            # of shared memory bytes required for LDGSTS.BYPASS                   
sm__sass_data_bytes_mem_shared_op_ldsm                                      Counter         byte            # of shared memory bytes required for LDSM                            
sm__sass_data_bytes_mem_shared_op_st                                        Counter         byte            # of shared memory bytes required for STS, ST                         
sm__sass_inst_executed                                                      Counter         inst            # of warp instructions executed                                       
sm__sass_inst_executed_memdesc_explicit                                     Counter         inst            # of warp instructions executed with explicit memory descriptor       
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_first                 Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                            policy on hit = EVICT_FIRST                                           
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_last                  Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                            policy on hit = EVICT_LAST                                            
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_normal                Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                            policy on hit = EVICT_NORMAL                                          
sm__sass_inst_executed_memdesc_explicit_hitprop_evict_normal_demote         Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                            policy on hit = EVICT_NORMAL_DEMOTE                                   
sm__sass_inst_executed_memdesc_explicit_missprop_evict_first                Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                            policy on miss = EVICT_FIRST                                          
sm__sass_inst_executed_memdesc_explicit_missprop_evict_normal               Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                            policy on miss = EVICT_NORMAL                                         
sm__sass_inst_executed_op_atom                                              Counter         inst            # of warp instructions executed: ATOM, ATOMG, ATOMS, ATOM             
sm__sass_inst_executed_op_branch                                            Counter         inst            # of warp instructions executed: BAR, JMP, BRX, BRXU, JMX, JMXU,      
                                                                                                            CALL, RET, WARPSYNC.EXCLUSIVE                                         
sm__sass_inst_executed_op_global                                            Counter         inst            # of warp instructions executed: LDG, STG, LD, ST, ATOM, ATOMG, RED   
sm__sass_inst_executed_op_global_atom                                         Counter         inst            # of warp instructions executed: ATOM, ATOMG                          
sm__sass_inst_executed_op_global_ld                                           Counter         inst            # of warp instructions executed: LDG, LD                              
sm__sass_inst_executed_op_global_red                                          Counter         inst            # of warp instructions executed: RED                                  
sm__sass_inst_executed_op_global_st                                           Counter         inst            # of warp instructions executed: STG, ST                              
sm__sass_inst_executed_op_ld                                                  Counter         inst            # of warp instructions executed: LDG, LD, LDS                         
sm__sass_inst_executed_op_ldgsts                                              Counter         inst            # of warp instructions executed: LDGSTS                               
sm__sass_inst_executed_op_ldgsts_cache_access                                 Counter         inst            # of warp instructions executed: LDGSTS.ACCESS                        
sm__sass_inst_executed_op_ldgsts_cache_bypass                                 Counter         inst            # of warp instructions executed: LDGSTS.BYPASS                        
sm__sass_inst_executed_op_ldsm                                                Counter         inst            # of warp instructions executed: LDSM                                 
sm__sass_inst_executed_op_local                                               Counter         inst            # of warp instructions executed: LDL, LD, STL, ST                     
sm__sass_inst_executed_op_local_ld                                            Counter         inst            # of warp instructions executed: LDL, LD                              
sm__sass_inst_executed_op_local_st                                            Counter         inst            # of warp instructions executed: STL, ST                              
sm__sass_inst_executed_op_memory_128b                                         Counter         inst            # of warp instructions executed by memory instructions with width 128 
                                                                                                              bit                                                                   
sm__sass_inst_executed_op_memory_16b                                          Counter         inst            # of warp instructions executed by memory instructions with width 16  
                                                                                                              bit                                                                   
sm__sass_inst_executed_op_memory_32b                                          Counter         inst            # of warp instructions executed by memory instructions with width 32  
                                                                                                              bit                                                                   
sm__sass_inst_executed_op_memory_64b                                          Counter         inst            # of warp instructions executed by memory instructions with width 64  
                                                                                                              bit                                                                   
sm__sass_inst_executed_op_memory_8b                                           Counter         inst            # of warp instructions executed by memory instructions with width 8   
                                                                                                              bit                                                                   
sm__sass_inst_executed_op_shared                                              Counter         inst            # of warp instructions executed: LDS, LD, STS, ST, ATOMS, ATOM        
sm__sass_inst_executed_op_shared_atom                                         Counter         inst            # of warp instructions executed: ATOMS ATOM                           
sm__sass_inst_executed_op_shared_ld                                           Counter         inst            # of warp instructions executed: LDS, LD                              
sm__sass_inst_executed_op_shared_st                                           Counter         inst            # of warp instructions executed: STS, ST                              
sm__sass_inst_executed_op_st                                                  Counter         inst            # of warp instructions executed: STG, ST, STL                         
sm__sass_inst_executed_op_texture                                             Counter         inst            # of warp instructions executed: texture                              
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts              Counter                         # of shared memory data bank conflicts generated by LDGSTS            
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts_cache_access Counter                         # of shared memory data bank conflicts generated by LDGSTS.ACCESS     
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts_cache_bypass Counter                         # of shared memory data bank conflicts generated by LDGSTS.BYPASS     
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm                Counter                         # of shared memory data bank conflicts generated by LDSM              
sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st                  Counter                         # of shared memory data bank conflicts generated by STS, ST           
sm__sass_l1tex_data_bank_writes_pipe_lsu_mem_shared_op_ldgsts_cache_access    Counter                         # of LDGSTS.ACCESS shared data bank writes                            
sm__sass_l1tex_data_bytes_write_pipe_lsu_mem_shared_op_ldgsts_cache_access    Counter         byte            # of LDGSTS.ACCESS shared data bytes write                            
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_atom                    Counter                         # of shared memory wavefronts processed by Data-Stage for ATOMS, ATOM 
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld                      Counter                         # of shared memory wavefronts processed by Data-Stage for LDS, LD     
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts                  Counter                         # of shared memory wavefronts processed by Data-Stage for LDGSTS      
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts_cache_access     Counter                         # of shared memory wavefronts processed by Data-Stage for             
                                                                                                              LDGSTS.ACCESS                                                         
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts_cache_bypass     Counter                         # of shared memory wavefronts processed by Data-Stage for             
                                                                                                              LDGSTS.BYPASS                                                         
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldsm                    Counter                         # of shared memory wavefronts processed by Data-Stage for LDSM        
sm__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_st                      Counter                         # of shared memory wavefronts processed by Data-Stage for STS, ST     
sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass      Counter         byte            # of bytes read from L2 into L1TEX M-Stage for LDGSTS.BYPASS          
sm__sass_l1tex_m_xbar2l1tex_read_sectors_mem_global_op_ldgsts_cache_bypass    Counter         sector          # of sectors read from L2 into L1TEX M-Stage for LDGSTS.BYPASS        
sm__sass_l1tex_pipe_lsu_wavefronts_mem_shared                                 Counter                         # of shared memory wavefronts processed by Data-Stage for LDS, LD,    
                                                                                                              STS, ST, ATOMS, ATOM, LDSM, LDGSTS                                    
sm__sass_l1tex_t_output_wavefronts_pipe_lsu_mem_global_op_ldgsts_cache_access Counter                         # of wavefronts sent to Data-Stage from T-Stage for LDGSTS.ACCESS     
sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts                       Counter         request         # of requests sent to T-Stage for LDGSTS                              
sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access          Counter         request         # of requests sent to T-Stage for LDGSTS.ACCESS                       
sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_bypass          Counter         request         # of requests sent to T-Stage for LDGSTS.BYPASS                       
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_atom                          Counter         sector          # of sectors requested for global atom                                
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ld                            Counter         sector          # of sectors requested for global loads                               
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts                        Counter         sector          # of sectors requested for LDGSTS                                     
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access           Counter         sector          # of sectors requested for LDGSTS.ACCESS                              
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_bypass           Counter         sector          # of sectors requested for LDGSTS.BYPASS                              
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_red                           Counter         sector          # of sectors requested for global reductions                          
sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_st                            Counter         sector          # of sectors requested for global stores                              
sm__sass_l1tex_t_sectors_pipe_lsu_mem_local_op_ld                             Counter         sector          # of sectors requested for local loads                                
sm__sass_l1tex_t_sectors_pipe_lsu_mem_local_op_st                             Counter         sector          # of sectors requested for local stores                               
sm__sass_l1tex_tags_mem_global                                                Counter                         # of L1 cache tag lookups generated by global memory instructions     
sm__sass_sectors_mem_global                                                   Counter         sector          # of global memory sectors accessed                                   
sm__sass_sectors_mem_local                                                    Counter         sector          # of local memory sectors accessed                                    
sm__sass_thread_inst_executed                                                 Counter         inst            # of thread instructions executed                                     
sm__sass_thread_inst_executed_op_bit_pred_on                                  Counter         inst            # of bit thread instructions executed where all predicates were true  
sm__sass_thread_inst_executed_op_control_pred_on                              Counter         inst            # of control-flow thread instructions executed where all predicates   
                                                                                                              were true                                                             
sm__sass_thread_inst_executed_op_conversion_pred_on                           Counter         inst            # of conversion thread instructions executed where all predicates     
                                                                                                              were true                                                             
sm__sass_thread_inst_executed_op_dadd_pred_on                                 Counter         inst            # of DADD thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_dfma_pred_on                                 Counter         inst            # of DFMA thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_dmul_pred_on                                 Counter         inst            # of DMUL thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_fadd_pred_on                                 Counter         inst            # of FADD thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_ffma_pred_on                                 Counter         inst            # of FFMA thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_fmul_pred_on                                 Counter         inst            # of FMUL thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_fp16_pred_on                                 Counter         inst            # of half-precision floating-point thread instructions executed where 
                                                                                                              all predicates were true                                              
sm__sass_thread_inst_executed_op_fp32_pred_on                                 Counter         inst            # of single-precision floating-point thread instructions executed     
                                                                                                              where all predicates were true                                        
sm__sass_thread_inst_executed_op_fp64_pred_on                                 Counter         inst            # of double-precision floating-point thread instructions executed     
                                                                                                              where all predicates were true                                        
sm__sass_thread_inst_executed_op_hadd_pred_on                                 Counter         inst            # of HADD thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_hfma_pred_on                                 Counter         inst            # of HFMA thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_hmul_pred_on                                 Counter         inst            # of HMUL thread instructions executed where all predicates were true 
sm__sass_thread_inst_executed_op_integer_pred_on                              Counter         inst            # of integer thread instructions executed where all predicates were   
                                                                                                              true                                                                  
sm__sass_thread_inst_executed_op_inter_thread_communication_pred_on           Counter         inst            # of inter-thread communication thread instructions executed where    
                                                                                                              all predicates were true                                              
sm__sass_thread_inst_executed_op_memory_pred_on                               Counter         inst            # of memory thread instructions executed where all predicates were    
                                                                                                              true                                                                  
sm__sass_thread_inst_executed_op_misc_pred_on                                 Counter         inst            # of miscellaneous instructions executed where all predicates were    
                                                                                                              true                                                                  
sm__sass_thread_inst_executed_op_uniform_pred_on                              Counter         inst            # of uniform thread instructions executed where all predicates were   
                                                                                                              true                                                                  
sm__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on                      Counter         inst            # of DADD, DMUL and DFMA thread instructions executed where all       
                                                                                                              predicates were true                                                  
sm__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on                      Counter         inst            # of FADD, FMUL and FFMA thread instructions executed where all       
                                                                                                              predicates were true                                                  
sm__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on                      Counter         inst            # of HADD, HMUL and HFMA thread instructions executed where all       
                                                                                                              predicates were true                                                  
sm__sass_thread_inst_executed_pred_on                                         Counter         inst            # of thread instructions where all predicates were true               
sm__thread_inst_executed_pipe_alu_pred_on                                     Counter         inst            # of thread instructions executed by alu where guard predicate was    
                                                                                                              true                                                                  
sm__thread_inst_executed_pipe_fma_pred_on                                     Counter         inst            # of thread instructions executed by fma where guard predicate was    
                                                                                                              true                                                                  
sm__thread_inst_executed_pipe_fp16_pred_on                                    Counter         inst            # of thread instructions executed by fp16 where guard predicate was   
                                                                                                              true                                                                  
sm__thread_inst_executed_pipe_fp64_pred_on                                    Counter         inst            # of thread instructions executed by fp64 where guard predicate was   
                                                                                                              true                                                                  
sm__thread_inst_executed_pipe_ipa_pred_on                                     Counter         inst            # of thread instructions executed by ipa where guard predicate was    
                                                                                                              true                                                                  
sm__thread_inst_executed_pipe_lsu_pred_on                                     Counter         inst            # of thread instructions executed by lsu where guard predicate was    
                                                                                                              true                                                                  
sm__thread_inst_executed_pipe_tensor_op_hmma_type_hfma2_pred_on               Counter         inst            # of thread instructions executed by tensor_op_hmma_type_hfma2 where  
                                                                                                              guard predicate was true                                              
sm__thread_inst_executed_pipe_xu_pred_on                                      Counter         inst            # of thread instructions executed by xu where guard predicate was true
sm__thread_inst_executed_pred_on_realtime                                     Counter         inst            # of thread instructions executed                                     
sm__threads_launched                                                          Counter         thread          # of threads launched                                                 
sm__threads_launched_shader_cs                                                Counter         thread          # of threads launched for CS                                          
sm__throughput                                                                Throughput      %               SM throughput assuming ideal load balancing across SMSPs              
sm__warps_active                                                              Counter         warp            cumulative # of warps in flight                                       
sm__warps_active_realtime                                                     Counter         warp            cumulative # of warps in flight                                       
sm__warps_active_shader_cs_realtime                                           Counter         warp            cumulative # of active CS warps                                       
sm__warps_launched                                                            Counter         warp            # of warps launched                                                   
sm__warps_launched_shader_cs                                                  Counter         warp            # of warps launched for CS                                            
smsp__amortized_warp_latency                                                  Ratio           cycle/warp      amortized warp duration (cycles)                                      
smsp__average_inst_executed_per_warp                                          Ratio           inst/warp       average # of instructions executed per warp                           
smsp__average_inst_executed_pipe_alu_per_warp                               Ratio           inst/warp       average # of instructions executed by pipe alu per warp               
smsp__average_inst_executed_pipe_cbu_per_warp                               Ratio           inst/warp       average # of instructions executed by pipe cbu per warp               
smsp__average_inst_executed_pipe_fma_per_warp                               Ratio           inst/warp       average # of instructions executed by pipe fma per warp               
smsp__average_inst_executed_pipe_fp16_per_warp                              Ratio           inst/warp       average # of instructions executed by pipe fp16 per warp              
smsp__average_inst_executed_pipe_fp64_per_warp                              Ratio           inst/warp       average # of instructions executed by pipe fp64 per warp              
smsp__average_inst_executed_pipe_ipa_per_warp                               Ratio           inst/warp       average # of instructions executed by pipe ipa per warp               
smsp__average_inst_executed_pipe_lsu_per_warp                               Ratio           inst/warp       average # of instructions executed by pipe lsu per warp               
smsp__average_inst_executed_pipe_tensor_op_dmma_per_warp                    Ratio           inst/warp       average # of instructions executed by pipe tensor_op_dmma per warp    
smsp__average_inst_executed_pipe_tensor_op_hmma_per_warp                    Ratio           inst/warp       average # of instructions executed by pipe tensor_op_hmma per warp    
smsp__average_inst_executed_pipe_tensor_op_hmma_type_hfma2_per_warp         Ratio           inst/warp       average # of instructions executed by pipe tensor_op_hmma_type_hfma2  
                                                                                                            per warp                                                              
smsp__average_inst_executed_pipe_tensor_op_imma_per_warp                    Ratio           inst/warp       average # of instructions executed by pipe tensor_op_imma per warp    
smsp__average_inst_executed_pipe_tex_per_warp                               Ratio           inst/warp       average # of instructions executed by pipe tex per warp               
smsp__average_inst_executed_pipe_uniform_per_warp                           Ratio           inst/warp       average # of instructions executed by pipe uniform per warp           
smsp__average_inst_executed_pipe_xu_per_warp                                Ratio           inst/warp       average # of instructions executed by pipe xu per warp                
smsp__average_threads_launched_per_warp                                     Ratio           thread/warp     average # of threads launched per warp                                
smsp__average_warp_latency                                                  Ratio           cycle           average warp duration (cycles)                                        
smsp__average_warp_latency_issue_stalled_barrier                            Ratio           inst/warp       average # of warp cycles spent waiting for sibling warps at a CTA     
                                                                                                            barrier                                                               
smsp__average_warp_latency_issue_stalled_branch_resolving                   Ratio           inst/warp       average # of warp cycles spent waiting for a branch target address to 
                                                                                                            be computed, and the warp PC to be updated                            
smsp__average_warp_latency_issue_stalled_dispatch_stall                     Ratio           inst/warp       average # of warp cycles spent waiting on a dispatch stall            
smsp__average_warp_latency_issue_stalled_drain                              Ratio           inst/warp       average # of warp cycles spent waiting after EXIT for all memory      
                                                                                                            instructions to complete so that warp resources can be freed          
smsp__average_warp_latency_issue_stalled_imc_miss                           Ratio           inst/warp       average # of warp cycles spent waiting for an immediate constant      
                                                                                                            cache (IMC) miss                                                      
smsp__average_warp_latency_issue_stalled_lg_throttle                        Ratio           inst/warp       average # of warp cycles spent waiting for a free entry in the LSU    
                                                                                                            instruction queue                                                     
smsp__average_warp_latency_issue_stalled_long_scoreboard                    Ratio           inst/warp       average # of warp cycles spent waiting for a scoreboard dependency on 
                                                                                                            L1TEX (local, global, surface, tex) operation                         
smsp__average_warp_latency_issue_stalled_math_pipe_throttle                 Ratio           inst/warp       average # of warp cycles spent waiting for an execution pipe to be    
                                                                                                            available                                                             
smsp__average_warp_latency_issue_stalled_membar                             Ratio           inst/warp       average # of warp cycles spent waiting on a memory barrier            
smsp__average_warp_latency_issue_stalled_mio_throttle                       Ratio           inst/warp       average # of warp cycles spent waiting for a free entry in the MIO    
                                                                                                            instruction queue                                                     
smsp__average_warp_latency_issue_stalled_misc                               Ratio           inst/warp       average # of warp cycles spent waiting on a miscellaneous hardware    
                                                                                                            reason                                                                
smsp__average_warp_latency_issue_stalled_no_instruction                     Ratio           inst/warp       average # of warp cycles spent waiting to be selected for instruction 
                                                                                                            fetch, or waiting on an instruction cache miss                        
smsp__average_warp_latency_issue_stalled_not_selected                       Ratio           inst/warp       average # of warp cycles spent waiting for the microscheduler to      
                                                                                                            select the warp to issue                                              
smsp__average_warp_latency_issue_stalled_selected                           Ratio           inst/warp       average # of warp cycles spent selected by the microscheduler to      
                                                                                                            issue an instruction                                                  
smsp__average_warp_latency_issue_stalled_short_scoreboard                   Ratio           inst/warp       average # of warp cycles spent waiting for a scoreboard dependency on 
                                                                                                            MIO operation other than (local, global, surface, tex)                
smsp__average_warp_latency_issue_stalled_sleeping                           Ratio           inst/warp       average # of warp cycles spent waiting for a nanosleep to expire      
smsp__average_warp_latency_issue_stalled_tex_throttle                       Ratio           inst/warp       average # of warp cycles spent waiting for a free entry in the TEX    
                                                                                                            instruction queue                                                     
smsp__average_warp_latency_issue_stalled_wait                               Ratio           inst/warp       average # of warp cycles spent waiting on a fixed latency execution   
                                                                                                            dependency                                                            
smsp__average_warp_latency_per_inst_executed                                Ratio           cycle           average # of cycles each warp was resident per instruction executed   
smsp__average_warp_latency_per_inst_issued                                  Ratio           cycle           average # of cycles each warp was resident per instruction issued     
smsp__average_warps_active_per_inst_executed                                Ratio           cycle           average # of cycles each warp was resident per instruction executed   
smsp__average_warps_active_per_issue_active                                 Ratio           warp            average # of warps resident per issue cycle                           
smsp__average_warps_issue_stalled_barrier_per_issue_active                  Ratio           inst            average # of warps resident per issue cycle, waiting for sibling      
                                                                                                            warps at a CTA barrier                                                
smsp__average_warps_issue_stalled_branch_resolving_per_issue_active         Ratio           inst            average # of warps resident per issue cycle, waiting for a branch     
                                                                                                            target address to be computed, and the warp PC to be updated          
smsp__average_warps_issue_stalled_dispatch_stall_per_issue_active           Ratio           inst            average # of warps resident per issue cycle, waiting on a dispatch    
                                                                                                            stall                                                                 
smsp__average_warps_issue_stalled_drain_per_issue_active                    Ratio           inst            average # of warps resident per issue cycle, waiting after EXIT for   
                                                                                                            all memory instructions to complete so that warp resources can be     
                                                                                                            freed                                                                 
smsp__average_warps_issue_stalled_imc_miss_per_issue_active                 Ratio           inst            average # of warps resident per issue cycle, waiting for an immediate 
                                                                                                            constant cache (IMC) miss                                             
smsp__average_warps_issue_stalled_lg_throttle_per_issue_active              Ratio           inst            average # of warps resident per issue cycle, waiting for a free entry 
                                                                                                            in the LSU instruction queue                                          
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active          Ratio           inst            average # of warps resident per issue cycle, waiting for a scoreboard 
                                                                                                            dependency on L1TEX (local, global, surface, tex) operation           
smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active       Ratio           inst            average # of warps resident per issue cycle, waiting for an execution 
                                                                                                            pipe to be available                                                  
smsp__average_warps_issue_stalled_membar_per_issue_active                   Ratio           inst            average # of warps resident per issue cycle, waiting on a memory      
                                                                                                            barrier                                                               
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active             Ratio           inst            average # of warps resident per issue cycle, waiting for a free entry 
                                                                                                            in the MIO instruction queue                                          
smsp__average_warps_issue_stalled_misc_per_issue_active                     Ratio           inst            average # of warps resident per issue cycle, waiting on a             
                                                                                                            miscellaneous hardware reason                                         
smsp__average_warps_issue_stalled_no_instruction_per_issue_active           Ratio           inst            average # of warps resident per issue cycle, waiting to be selected   
                                                                                                            for instruction fetch, or waiting on an instruction cache miss        
smsp__average_warps_issue_stalled_not_selected_per_issue_active             Ratio           inst            average # of warps resident per issue cycle, waiting for the          
                                                                                                            microscheduler to select the warp to issue                            
smsp__average_warps_issue_stalled_selected_per_issue_active                 Ratio           inst            average # of warps resident per issue cycle, selected by the          
                                                                                                            microscheduler to issue an instruction                                
smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active         Ratio           inst            average # of warps resident per issue cycle, waiting for a scoreboard 
                                                                                                            dependency on MIO operation other than (local, global, surface, tex)  
smsp__average_warps_issue_stalled_sleeping_per_issue_active                 Ratio           inst            average # of warps resident per issue cycle, waiting for a nanosleep  
                                                                                                            to expire                                                             
smsp__average_warps_issue_stalled_tex_throttle_per_issue_active             Ratio           inst            average # of warps resident per issue cycle, waiting for a free entry 
                                                                                                            in the TEX instruction queue                                          
smsp__average_warps_issue_stalled_wait_per_issue_active                     Ratio           inst            average # of warps resident per issue cycle, waiting on a fixed       
                                                                                                            latency execution dependency                                          
smsp__cycles_active                                                         Counter         cycle           # of cycles with at least one warp in flight                          
smsp__cycles_elapsed                                                        Counter         cycle           # of cycles elapsed on SMSP                                           
smsp__cycles_in_frame                                                       Counter         cycle           # of cycles in user-defined frame                                     
smsp__cycles_in_region                                                      Counter         cycle           # of cycles in user-defined region                                    
smsp__inst_executed                                                         Counter         inst            # of warp instructions executed                                       
smsp__inst_executed_op_branch                                               Counter         inst            # of warp instructions executed: by cbu pipe except BMOV, BSSY        
smsp__inst_executed_op_generic_atom                                         Counter         inst            # of warp instructions executed: ATOM.*                               
smsp__inst_executed_op_generic_atom_dot_alu                                 Counter         inst            # of warp instructions executed: ATOM.ALU, ATOMG.ALU (non-CAS)        
smsp__inst_executed_op_generic_atom_dot_alu_pred_off_all                    Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                            ATOM.ALU, ATOMG.ALU (non-CAS)                                         
smsp__inst_executed_op_generic_atom_dot_alu_pred_on_any                     Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            ATOM.ALU, ATOMG.ALU (non-CAS)                                         
smsp__inst_executed_op_generic_atom_dot_cas                                 Counter         inst            # of warp instructions executed: ATOM.CAS, ATOMG.CAS                  
smsp__inst_executed_op_generic_atom_dot_cas_pred_off_all                    Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                            ATOM.CAS, ATOMG.CAS                                                   
smsp__inst_executed_op_generic_atom_dot_cas_pred_on_any                     Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            ATOM.CAS, ATOMG.CAS                                                   
smsp__inst_executed_op_generic_atom_pred_off_all                            Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                            ATOM.*                                                                
smsp__inst_executed_op_generic_atom_pred_on_any                             Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            ATOM.*                                                                
smsp__inst_executed_op_generic_ld                                           Counter         inst            # of warp instructions executed: LD                                   
smsp__inst_executed_op_generic_ld_pred_off_all                              Counter         inst            # of warp instructions executed with all threads predicated off: LD   
smsp__inst_executed_op_generic_ld_pred_on_any                               Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            LD                                                                    
smsp__inst_executed_op_generic_st                                           Counter         inst            # of warp instructions executed: ST                                   
smsp__inst_executed_op_generic_st_pred_off_all                              Counter         inst            # of warp instructions executed with all threads predicated off: ST   
smsp__inst_executed_op_generic_st_pred_on_any                               Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            ST                                                                    
smsp__inst_executed_op_global_ld                                            Counter         inst            # of warp instructions executed: LDG                                  
smsp__inst_executed_op_global_ld_pred_off_all                               Counter         inst            # of warp instructions executed with all threads predicated off: LDG  
smsp__inst_executed_op_global_ld_pred_on_any                                Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            LDG                                                                   
smsp__inst_executed_op_global_red                                           Counter         inst            # of warp instructions executed: RED                                  
smsp__inst_executed_op_global_red_pred_off_all                              Counter         inst            # of warp instructions executed with all threads predicated off: RED  
smsp__inst_executed_op_global_red_pred_on_any                               Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            RED                                                                   
smsp__inst_executed_op_global_st                                            Counter         inst            # of warp instructions executed: STG                                  
smsp__inst_executed_op_global_st_pred_off_all                               Counter         inst            # of warp instructions executed with all threads predicated off: STG  
smsp__inst_executed_op_global_st_pred_on_any                                Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            STG                                                                   
smsp__inst_executed_op_ldc                                                  Counter         inst            # of warp instructions executed: LDC                                  
smsp__inst_executed_op_ldc_pred_off_all                                     Counter         inst            # of warp instructions executed with all threads predicated off: LDC  
smsp__inst_executed_op_ldc_pred_on_any                                      Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            LDC                                                                   
smsp__inst_executed_op_ldgsts                                               Counter         inst            # of warp instructions executed: LDGSTS                               
smsp__inst_executed_op_ldgsts_pred_off_all                                  Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                            LDGSTS                                                                
smsp__inst_executed_op_ldgsts_pred_on_any                                   Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            LDGSTS                                                                
smsp__inst_executed_op_ldsm                                                 Counter         inst            # of warp instructions executed: LDSM                                 
smsp__inst_executed_op_ldsm_pred_off_all                                    Counter         inst            # of warp instructions executed with all threads predicated off: LDSM 
smsp__inst_executed_op_ldsm_pred_on_any                                     Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            LDSM                                                                  
smsp__inst_executed_op_local_ld                                             Counter         inst            # of warp instructions executed: LDL                                  
smsp__inst_executed_op_local_ld_pred_off_all                                Counter         inst            # of warp instructions executed with all threads predicated off: LDL  
smsp__inst_executed_op_local_ld_pred_on_any                                 Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                            LDL                                                                   
smsp__inst_executed_op_local_st                                             Counter         inst            # of warp instructions executed: STL                                  
smsp__inst_executed_op_local_st_pred_off_all                                Counter         inst            # of warp instructions executed with all threads predicated off: STL  
smsp__inst_executed_op_local_st_pred_on_any                                   Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              STL                                                                   
smsp__inst_executed_op_shared_atom                                            Counter         inst            # of warp instructions executed: ATOMS.*                              
smsp__inst_executed_op_shared_atom_dot_alu                                    Counter         inst            # of warp instructions executed: ATOMS.ALU (non-CAS)                  
smsp__inst_executed_op_shared_atom_dot_alu_pred_off_all                       Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                              ATOMS.ALU (non-CAS)                                                   
smsp__inst_executed_op_shared_atom_dot_alu_pred_on_any                        Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              ATOMS.ALU (non-CAS)                                                   
smsp__inst_executed_op_shared_atom_dot_cas                                    Counter         inst            # of warp instructions executed: ATOMS.CAS                            
smsp__inst_executed_op_shared_atom_dot_cas_pred_off_all                       Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                              ATOMS.CAS                                                             
smsp__inst_executed_op_shared_atom_dot_cas_pred_on_any                        Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              ATOMS.CAS                                                             
smsp__inst_executed_op_shared_atom_pred_off_all                               Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                              ATOMS.*                                                               
smsp__inst_executed_op_shared_atom_pred_on_any                                Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              ATOMS.*                                                               
smsp__inst_executed_op_shared_ld                                              Counter         inst            # of warp instructions executed: LDS                                  
smsp__inst_executed_op_shared_ld_pred_off_all                                 Counter         inst            # of warp instructions executed with all threads predicated off: LDS  
smsp__inst_executed_op_shared_ld_pred_on_any                                  Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              LDS                                                                   
smsp__inst_executed_op_shared_st                                              Counter         inst            # of warp instructions executed: STS                                  
smsp__inst_executed_op_shared_st_pred_off_all                                 Counter         inst            # of warp instructions executed with all threads predicated off: STS  
smsp__inst_executed_op_shared_st_pred_on_any                                  Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              STS                                                                   
smsp__inst_executed_op_surface_atom                                           Counter         inst            # of warp instructions executed: SUATOM.*                             
smsp__inst_executed_op_surface_atom_dot_alu                                   Counter         inst            # of warp instructions executed: SUATOM.ALU (non-CAS)                 
smsp__inst_executed_op_surface_atom_dot_alu_pred_off_all                      Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                              SUATOM.ALU (non-CAS)                                                  
smsp__inst_executed_op_surface_atom_dot_alu_pred_on_any                       Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              SUATOM.ALU (non-CAS)                                                  
smsp__inst_executed_op_surface_atom_dot_cas                                   Counter         inst            # of warp instructions executed: SUATOM.CAS                           
smsp__inst_executed_op_surface_atom_dot_cas_pred_off_all                      Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                              SUATOM.CAS                                                            
smsp__inst_executed_op_surface_atom_dot_cas_pred_on_any                       Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              SUATOM.CAS                                                            
smsp__inst_executed_op_surface_atom_pred_off_all                              Counter         inst            # of warp instructions executed with all threads predicated off:      
                                                                                                              SUATOM.*                                                              
smsp__inst_executed_op_surface_atom_pred_on_any                               Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              SUATOM.*                                                              
smsp__inst_executed_op_surface_ld                                             Counter         inst            # of warp instructions executed: SULD                                 
smsp__inst_executed_op_surface_ld_pred_off_all                                Counter         inst            # of warp instructions executed with all threads predicated off: SULD 
smsp__inst_executed_op_surface_ld_pred_on_any                                 Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              SULD                                                                  
smsp__inst_executed_op_surface_red                                            Counter         inst            # of warp instructions executed: SURED                                
smsp__inst_executed_op_surface_red_pred_off_all                               Counter         inst            # of warp instructions executed with all threads predicated off: SURED
smsp__inst_executed_op_surface_red_pred_on_any                                Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              SURED                                                                 
smsp__inst_executed_op_surface_st                                             Counter         inst            # of warp instructions executed: SUST                                 
smsp__inst_executed_op_surface_st_pred_off_all                                Counter         inst            # of warp instructions executed with all threads predicated off: SUST 
smsp__inst_executed_op_surface_st_pred_on_any                                 Counter         inst            # of warp instructions executed with at least 1 thread predicated on: 
                                                                                                              SUST                                                                  
smsp__inst_executed_op_texture                                                Counter         inst            # of warp instructions executed: texture                              
smsp__inst_executed_pipe_adu                                                  Counter         inst            # of warp instructions executed by adu pipe                           
smsp__inst_executed_pipe_alu                                                  Counter         inst            # of warp instructions executed by alu pipe                           
smsp__inst_executed_pipe_cbu                                                  Counter         inst            # of warp instructions executed by cbu pipe                           
smsp__inst_executed_pipe_cbu_pred_off_all                                     Counter         inst            # of warp instructions executed by cbu pipe with all threads          
                                                                                                              predicated off                                                        
smsp__inst_executed_pipe_cbu_pred_on_any                                      Counter         inst            # of warp instructions executed by cbu pipe with at least 1 thread    
                                                                                                              predicated on                                                         
smsp__inst_executed_pipe_fma                                                  Counter         inst            # of warp instructions executed by fma pipe                           
smsp__inst_executed_pipe_fp16                                                 Counter         inst            # of warp instructions executed by fp16 pipe                          
smsp__inst_executed_pipe_fp64                                                 Counter         inst            # of warp instructions executed by fp64 pipe                          
smsp__inst_executed_pipe_ipa                                                  Counter         inst            # of warp instructions executed by ipa pipe                           
smsp__inst_executed_pipe_lsu                                                  Counter         inst            # of warp instructions executed by lsu pipe                           
smsp__inst_executed_pipe_tensor                                               Counter         inst            # of warp instructions executed by tensor pipe                        
smsp__inst_executed_pipe_tensor_op_dmma                                       Counter         inst            # of warp instructions executed by tensor pipe (DMMA ops)             
smsp__inst_executed_pipe_tensor_op_hmma                                       Counter         inst            # of warp instructions executed by tensor pipe (HMMA ops)             
smsp__inst_executed_pipe_tensor_op_hmma_type_hfma2                            Counter         inst            # of warp instructions executed by tensor pipe (HFMA2.MMA ops)        
smsp__inst_executed_pipe_tensor_op_imma                                       Counter         inst            # of warp instructions executed by tensor pipe (IMMA ops)             
smsp__inst_executed_pipe_tex                                                  Counter         inst            # of warp instructions executed by tex pipe                           
smsp__inst_executed_pipe_uniform                                              Counter         inst            # of warp instructions executed by uniform pipe                       
smsp__inst_executed_pipe_xu                                                   Counter         inst            # of warp instructions executed by xu pipe                            
smsp__inst_executed_shader_cs                                                 Counter         inst            # of warp instructions executed by CS                                 
smsp__inst_issued                                                             Counter         inst            # of warp instructions issued                                         
smsp__inst_issued_per_issue_active                                            Ratio           inst/cycle      # of warp instructions per issue cycle                                
smsp__issue_active                                                            Counter         cycle           # of cycles where 1 instruction was issued                            
smsp__issue_inst0                                                             Counter         cycle           # of active cycles where 0 instructions were issued                   
smsp__issue_inst1                                                             Counter         cycle           # of active cycles where 1 instruction was issued                     
smsp__l1tex_lsuin_requests                                                    Counter         request         # of local/global/shared/attribute/S2R/SHFL instructions sent to LSU  
smsp__mio2rf_writeback_active                                                 Counter         cycle           # of cycles where the MIO to register file writeback interface was    
                                                                                                              active                                                                
smsp__mio2rf_writeback_active_pipe_adu                                        Counter         cycle           # of cycles where the MIO to register file writeback interface was    
                                                                                                              active for ADU return data                                            
smsp__mio2rf_writeback_active_pipe_lsu                                        Counter         cycle           # of cycles where the MIO to register file writeback interface was    
                                                                                                              active for LSU return data                                            
smsp__mio2rf_writeback_active_pipe_tex                                        Counter         cycle           # of cycles where the MIO to register file writeback interface was    
                                                                                                              active for TEX return data                                            
smsp__pipe_alu_cycles_active                                                  Counter         cycle           # of cycles where alu pipe was active                                 
smsp__pipe_fma_cycles_active                                                  Counter         cycle           # of cycles where fma pipe was active                                 
smsp__pipe_fp64_cycles_active                                                 Counter         cycle           # of cycles where fp64 pipe was active                                
smsp__pipe_shared_cycles_active                                               Counter         cycle           # of cycles where the shared pipe (fp64+tensor) was active            
smsp__pipe_tensor_cycles_active                                               Counter         cycle           # of cycles where tensor pipe was active                              
smsp__pipe_tensor_op_dmma_cycles_active                                       Counter         inst            # of cycles where tensor pipe was active (DMMA ops)                   
smsp__pipe_tensor_op_hmma_cycles_active                                       Counter         cycle           # of cycles where tensor pipe was active (HMMA ops)                   
smsp__pipe_tensor_op_imma_cycles_active                                       Counter         cycle           # of cycles where tensor pipe was active (IMMA ops)                   
smsp__sass_average_branch_targets_threads_uniform                             Ratio                           proportion of branch targets where all active threads selected the    
                                                                                                              same branch target                                                    
smsp__sass_average_data_bytes_per_sector_mem_global                           Ratio           byte/sector     average # of bytes per sector by global memory operations             
smsp__sass_average_data_bytes_per_sector_mem_global_op_atom                   Ratio           byte/sector     average # of bytes per sector by global atom                          
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld                     Ratio           byte/sector     average # of bytes per sector by global loads                         
smsp__sass_average_data_bytes_per_sector_mem_global_op_ldgsts                 Ratio           byte/sector     average # of bytes per sector by LDGSTS global loads                  
smsp__sass_average_data_bytes_per_sector_mem_global_op_ldgsts_cache_access    Ratio           byte/sector     average # of bytes per sector by LDGSTS.ACCESS global loads           
smsp__sass_average_data_bytes_per_sector_mem_global_op_ldgsts_cache_bypass    Ratio           byte/sector     average # of bytes per sector by LDGSTS.BYPASS global loads           
smsp__sass_average_data_bytes_per_sector_mem_global_op_red                    Ratio           byte/sector     average # of bytes per sector by global reductions                    
smsp__sass_average_data_bytes_per_sector_mem_global_op_st                     Ratio           byte/sector     average # of bytes per sector by global stores                        
smsp__sass_average_data_bytes_per_sector_mem_local                            Ratio           byte/sector     average # of bytes per sector by local memory operations              
smsp__sass_average_data_bytes_per_sector_mem_local_op_ld                      Ratio           byte/sector     average # of bytes per sector by local loads                          
smsp__sass_average_data_bytes_per_sector_mem_local_op_st                      Ratio           byte/sector     average # of bytes per sector by local stores                         
smsp__sass_average_data_bytes_per_wavefront_mem_shared                        Ratio                           average # of shared memory bytes per wavefront required by LDS, LD,   
                                                                                                              STS, ST, ATOMS, ATOM, LDSM, LDGSTS                                    
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_atom                Ratio                           average # of shared memory bytes per wavefront required by ATOMS, ATOM
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ld                  Ratio                           average # of shared memory bytes per wavefront required by LDS, LD    
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ldgsts              Ratio                           average # of shared memory bytes per wavefront required by LDGSTS     
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ldgsts_cache_access Ratio                           average # of shared memory bytes per wavefront required by            
                                                                                                              LDGSTS.ACCESS                                                         
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ldgsts_cache_bypass Ratio                           average # of shared memory bytes per wavefront required by            
                                                                                                              LDGSTS.BYPASS                                                         
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_ldsm                Ratio                           average # of shared memory bytes per wavefront required by LDSM       
smsp__sass_average_data_bytes_per_wavefront_mem_shared_op_st                  Ratio                           average # of shared memory bytes per wavefront required by STS, ST    
smsp__sass_branch_targets                                                     Counter                         # of unique branch targets assigned to the PC                         
smsp__sass_branch_targets_threads_divergent                                   Counter                         incremented only when there are two or more active threads with       
                                                                                                              different branch target                                               
smsp__sass_branch_targets_threads_uniform                                     Counter                         # of branch executions where all active threads selected the same     
                                                                                                              branch target                                                         
smsp__sass_data_bytes_mem_global                                              Counter         byte            # of bytes required for global operations                             
smsp__sass_data_bytes_mem_global_op_atom                                      Counter         byte            # of bytes required for global atom                                   
smsp__sass_data_bytes_mem_global_op_ld                                        Counter         byte            # of bytes required for global loads                                  
smsp__sass_data_bytes_mem_global_op_ldgsts                                    Counter         byte            # of bytes required for LDGSTS global loads                           
smsp__sass_data_bytes_mem_global_op_ldgsts_cache_access                       Counter         byte            # of bytes required for LDGSTS.ACCESS global loads                    
smsp__sass_data_bytes_mem_global_op_ldgsts_cache_bypass                                   Counter         byte            # of bytes required for LDGSTS.BYPASS global loads                    
smsp__sass_data_bytes_mem_global_op_red                                                   Counter         byte            # of bytes required for global reductions                             
smsp__sass_data_bytes_mem_global_op_st                                                    Counter         byte            # of bytes required for global stores                                 
smsp__sass_data_bytes_mem_local                                                           Counter         byte            # of bytes required for local operations                              
smsp__sass_data_bytes_mem_local_op_ld                                                     Counter         byte            # of bytes required for local loads                                   
smsp__sass_data_bytes_mem_local_op_st                                                     Counter         byte            # of bytes required for local stores                                  
smsp__sass_data_bytes_mem_shared                                                          Counter         byte            # of shared memory bytes required for LDS, LD, STS, ST, ATOMS, ATOM,  
                                                                                                                          LDSM, LDGSTS                                                          
smsp__sass_data_bytes_mem_shared_op_atom                                                  Counter         byte            # of shared memory bytes required for ATOMS, ATOM                     
smsp__sass_data_bytes_mem_shared_op_ld                                                    Counter         byte            # of shared memory bytes required for LDS, LD                         
smsp__sass_data_bytes_mem_shared_op_ldgsts                                                Counter         byte            # of shared memory bytes required for LDGSTS                          
smsp__sass_data_bytes_mem_shared_op_ldgsts_cache_access                                   Counter         byte            # of shared memory bytes required for LDGSTS.ACCESS                   
smsp__sass_data_bytes_mem_shared_op_ldgsts_cache_bypass                                   Counter         byte            # of shared memory bytes required for LDGSTS.BYPASS                   
smsp__sass_data_bytes_mem_shared_op_ldsm                                                  Counter         byte            # of shared memory bytes required for LDSM                            
smsp__sass_data_bytes_mem_shared_op_st                                                    Counter         byte            # of shared memory bytes required for STS, ST                         
smsp__sass_inst_executed                                                                  Counter         inst            # of warp instructions executed                                       
smsp__sass_inst_executed_memdesc_explicit                                                 Counter         inst            # of warp instructions executed with explicit memory descriptor       
smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_first                             Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                                          policy on hit = EVICT_FIRST                                           
smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_last                              Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                                          policy on hit = EVICT_LAST                                            
smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_normal                            Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                                          policy on hit = EVICT_NORMAL                                          
smsp__sass_inst_executed_memdesc_explicit_hitprop_evict_normal_demote                     Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                                          policy on hit = EVICT_NORMAL_DEMOTE                                   
smsp__sass_inst_executed_memdesc_explicit_missprop_evict_first                            Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                                          policy on miss = EVICT_FIRST                                          
smsp__sass_inst_executed_memdesc_explicit_missprop_evict_normal                           Counter         inst            # of warp instructions executed with explicit memory descriptor's     
                                                                                                                          policy on miss = EVICT_NORMAL                                         
smsp__sass_inst_executed_op_atom                                                          Counter         inst            # of warp instructions executed: ATOM, ATOMG, ATOMS, ATOM             
smsp__sass_inst_executed_op_branch                                                        Counter         inst            # of warp instructions executed: BAR, JMP, BRX, BRXU, JMX, JMXU,      
                                                                                                                          CALL, RET, WARPSYNC.EXCLUSIVE                                         
smsp__sass_inst_executed_op_global                                                        Counter         inst            # of warp instructions executed: LDG, STG, LD, ST, ATOM, ATOMG, RED   
smsp__sass_inst_executed_op_global_atom                                                   Counter         inst            # of warp instructions executed: ATOM, ATOMG                          
smsp__sass_inst_executed_op_global_ld                                                     Counter         inst            # of warp instructions executed: LDG, LD                              
smsp__sass_inst_executed_op_global_red                                                    Counter         inst            # of warp instructions executed: RED                                  
smsp__sass_inst_executed_op_global_st                                                     Counter         inst            # of warp instructions executed: STG, ST                              
smsp__sass_inst_executed_op_ld                                                            Counter         inst            # of warp instructions executed: LDG, LD, LDS                         
smsp__sass_inst_executed_op_ldgsts                                                        Counter         inst            # of warp instructions executed: LDGSTS                               
smsp__sass_inst_executed_op_ldgsts_cache_access                                           Counter         inst            # of warp instructions executed: LDGSTS.ACCESS                        
smsp__sass_inst_executed_op_ldgsts_cache_bypass                                           Counter         inst            # of warp instructions executed: LDGSTS.BYPASS                        
smsp__sass_inst_executed_op_ldsm                                                          Counter         inst            # of warp instructions executed: LDSM                                 
smsp__sass_inst_executed_op_local                                                         Counter         inst            # of warp instructions executed: LDL, LD, STL, ST                     
smsp__sass_inst_executed_op_local_ld                                                      Counter         inst            # of warp instructions executed: LDL, LD                              
smsp__sass_inst_executed_op_local_st                                                      Counter         inst            # of warp instructions executed: STL, ST                              
smsp__sass_inst_executed_op_memory_128b                                                   Counter         inst            # of warp instructions executed by memory instructions with width 128 
                                                                                                                          bit                                                                   
smsp__sass_inst_executed_op_memory_16b                                                    Counter         inst            # of warp instructions executed by memory instructions with width 16  
                                                                                                                          bit                                                                   
smsp__sass_inst_executed_op_memory_32b                                                    Counter         inst            # of warp instructions executed by memory instructions with width 32  
                                                                                                                          bit                                                                   
smsp__sass_inst_executed_op_memory_64b                                                    Counter         inst            # of warp instructions executed by memory instructions with width 64  
                                                                                                                          bit                                                                   
smsp__sass_inst_executed_op_memory_8b                                                     Counter         inst            # of warp instructions executed by memory instructions with width 8   
                                                                                                                          bit                                                                   
smsp__sass_inst_executed_op_shared                                                        Counter         inst            # of warp instructions executed: LDS, LD, STS, ST, ATOMS, ATOM        
smsp__sass_inst_executed_op_shared_atom                                                   Counter         inst            # of warp instructions executed: ATOMS ATOM                           
smsp__sass_inst_executed_op_shared_ld                                                     Counter         inst            # of warp instructions executed: LDS, LD                              
smsp__sass_inst_executed_op_shared_st                                                     Counter         inst            # of warp instructions executed: STS, ST                              
smsp__sass_inst_executed_op_st                                                            Counter         inst            # of warp instructions executed: STG, ST, STL                         
smsp__sass_inst_executed_op_texture                                                       Counter         inst            # of warp instructions executed: texture                              
smsp__sass_l1tex_average_t_sectors_per_request_pipe_lsu_mem_global_op_ldgsts              Ratio           sector/request  average # of sectors requested per request sent to T stage by LDGSTS  
smsp__sass_l1tex_average_t_sectors_per_request_pipe_lsu_mem_global_op_ldgsts_cache_access Ratio           sector/request  average # of sectors requested per request sent to T stage by         
                                                                                                                          LDGSTS.ACCESS                                                         
smsp__sass_l1tex_average_t_sectors_per_request_pipe_lsu_mem_global_op_ldgsts_cache_bypass Ratio           sector/request  average # of sectors requested per request sent to T stage by         
                                                                                                                          LDGSTS.BYPASS                                                         
smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts                        Counter                         # of shared memory data bank conflicts generated by LDGSTS            
smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts_cache_access           Counter                         # of shared memory data bank conflicts generated by LDGSTS.ACCESS     
smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts_cache_bypass           Counter                         # of shared memory data bank conflicts generated by LDGSTS.BYPASS     
smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm                          Counter                         # of shared memory data bank conflicts generated by LDSM              
smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st                            Counter                         # of shared memory data bank conflicts generated by STS, ST           
smsp__sass_l1tex_data_bank_writes_pipe_lsu_mem_shared_op_ldgsts_cache_access              Counter                         # of LDGSTS.ACCESS shared data bank writes                            
smsp__sass_l1tex_data_bytes_write_pipe_lsu_mem_shared_op_ldgsts_cache_access              Counter         byte            # of LDGSTS.ACCESS shared data bytes write                            
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_atom                              Counter                         # of shared memory wavefronts processed by Data-Stage for ATOMS, ATOM 
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ld                                Counter                         # of shared memory wavefronts processed by Data-Stage for LDS, LD     
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts                            Counter                         # of shared memory wavefronts processed by Data-Stage for LDGSTS      
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts_cache_access               Counter                         # of shared memory wavefronts processed by Data-Stage for             
                                                                                                                          LDGSTS.ACCESS                                                         
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldgsts_cache_bypass               Counter                         # of shared memory wavefronts processed by Data-Stage for             
                                                                                                                          LDGSTS.BYPASS                                                         
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_ldsm                              Counter                         # of shared memory wavefronts processed by Data-Stage for LDSM        
smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_st                                Counter                         # of shared memory wavefronts processed by Data-Stage for STS, ST     
smsp__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass                Counter         byte            # of bytes read from L2 into L1TEX M-Stage for LDGSTS.BYPASS          
smsp__sass_l1tex_m_xbar2l1tex_read_sectors_mem_global_op_ldgsts_cache_bypass              Counter         sector          # of sectors read from L2 into L1TEX M-Stage for LDGSTS.BYPASS        
smsp__sass_l1tex_pipe_lsu_wavefronts_mem_shared                                           Counter                         # of shared memory wavefronts processed by Data-Stage for LDS, LD,    
                                                                                                                          STS, ST, ATOMS, ATOM, LDSM, LDGSTS                                    
smsp__sass_l1tex_t_output_wavefronts_pipe_lsu_mem_global_op_ldgsts_cache_access           Counter                         # of wavefronts sent to Data-Stage from T-Stage for LDGSTS.ACCESS     
smsp__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts                                 Counter         request         # of requests sent to T-Stage for LDGSTS                              
smsp__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access                    Counter         request         # of requests sent to T-Stage for LDGSTS.ACCESS                       
smsp__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_bypass                    Counter         request         # of requests sent to T-Stage for LDGSTS.BYPASS                       
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_atom                                    Counter         sector          # of sectors requested for global atom                                
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ld                                      Counter         sector          # of sectors requested for global loads                               
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts                                  Counter         sector          # of sectors requested for LDGSTS                                     
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access                     Counter         sector          # of sectors requested for LDGSTS.ACCESS                              
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_bypass                     Counter         sector          # of sectors requested for LDGSTS.BYPASS                              
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_red                                     Counter         sector          # of sectors requested for global reductions                          
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_st                                      Counter         sector          # of sectors requested for global stores                              
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_local_op_ld                                       Counter         sector          # of sectors requested for local loads                                
smsp__sass_l1tex_t_sectors_pipe_lsu_mem_local_op_st                                       Counter         sector          # of sectors requested for local stores                               
smsp__sass_l1tex_tags_mem_global                                                          Counter                         # of L1 cache tag lookups generated by global memory instructions     
smsp__sass_sectors_mem_global                                                             Counter         sector          # of global memory sectors accessed                                   
smsp__sass_sectors_mem_local                                                              Counter         sector          # of local memory sectors accessed                                    
smsp__sass_thread_inst_executed                                                           Counter         inst            # of thread instructions executed                                     
smsp__sass_thread_inst_executed_op_bit_pred_on                                            Counter         inst            # of bit thread instructions executed where all predicates were true  
smsp__sass_thread_inst_executed_op_control_pred_on                                        Counter         inst            # of control-flow thread instructions executed where all predicates   
                                                                                                                          were true                                                             
smsp__sass_thread_inst_executed_op_conversion_pred_on                                     Counter         inst            # of conversion thread instructions executed where all predicates     
                                                                                                                          were true                                                             
smsp__sass_thread_inst_executed_op_dadd_pred_on                                           Counter         inst            # of DADD thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_dfma_pred_on                                           Counter         inst            # of DFMA thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_dmul_pred_on                                           Counter         inst            # of DMUL thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_fadd_pred_on                                           Counter         inst            # of FADD thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_ffma_pred_on                                           Counter         inst            # of FFMA thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_fmul_pred_on                                           Counter         inst            # of FMUL thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_fp16_pred_on                                           Counter         inst            # of half-precision floating-point thread instructions executed where 
                                                                                                                          all predicates were true                                              
smsp__sass_thread_inst_executed_op_fp32_pred_on                                           Counter         inst            # of single-precision floating-point thread instructions executed     
                                                                                                                          where all predicates were true                                        
smsp__sass_thread_inst_executed_op_fp64_pred_on                                           Counter         inst            # of double-precision floating-point thread instructions executed     
                                                                                                                          where all predicates were true                                        
smsp__sass_thread_inst_executed_op_hadd_pred_on                                           Counter         inst            # of HADD thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_hfma_pred_on                                           Counter         inst            # of HFMA thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_hmul_pred_on                                           Counter         inst            # of HMUL thread instructions executed where all predicates were true 
smsp__sass_thread_inst_executed_op_integer_pred_on                          Counter         inst            # of integer thread instructions executed where all predicates were   
                                                                                                            true                                                                  
smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on       Counter         inst            # of inter-thread communication thread instructions executed where    
                                                                                                            all predicates were true                                              
smsp__sass_thread_inst_executed_op_memory_pred_on                           Counter         inst            # of memory thread instructions executed where all predicates were    
                                                                                                            true                                                                  
smsp__sass_thread_inst_executed_op_misc_pred_on                             Counter         inst            # of miscellaneous instructions executed where all predicates were    
                                                                                                            true                                                                  
smsp__sass_thread_inst_executed_op_uniform_pred_on                          Counter         inst            # of uniform thread instructions executed where all predicates were   
                                                                                                            true                                                                  
smsp__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on                  Counter         inst            # of DADD, DMUL and DFMA thread instructions executed where all       
                                                                                                            predicates were true                                                  
smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on                  Counter         inst            # of FADD, FMUL and FFMA thread instructions executed where all       
                                                                                                            predicates were true                                                  
smsp__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on                  Counter         inst            # of HADD, HMUL and HFMA thread instructions executed where all       
                                                                                                            predicates were true                                                  
smsp__sass_thread_inst_executed_pred_on                                     Counter         inst            # of thread instructions where all predicates were true               
smsp__tex_requests                                                          Counter         cycle           # of TEX requests sent from SMSP                                      
smsp__thread_inst_executed                                                  Counter         inst            # of thread instructions executed                                     
smsp__thread_inst_executed_per_inst_executed                                Ratio                           average # of active threads per instruction executed                  
smsp__thread_inst_executed_pipe_alu_pred_on                                 Counter         inst            # of thread instructions executed by alu where guard predicate was    
                                                                                                            true                                                                  
smsp__thread_inst_executed_pipe_fma_pred_on                                 Counter         inst            # of thread instructions executed by fma where guard predicate was    
                                                                                                            true                                                                  
smsp__thread_inst_executed_pipe_fp16_pred_on                                Counter         inst            # of thread instructions executed by fp16 where guard predicate was   
                                                                                                            true                                                                  
smsp__thread_inst_executed_pipe_fp64_pred_on                                Counter         inst            # of thread instructions executed by fp64 where guard predicate was   
                                                                                                            true                                                                  
smsp__thread_inst_executed_pipe_ipa_pred_on                                 Counter         inst            # of thread instructions executed by ipa where guard predicate was    
                                                                                                            true                                                                  
smsp__thread_inst_executed_pipe_lsu_pred_on                                 Counter         inst            # of thread instructions executed by lsu where guard predicate was    
                                                                                                            true                                                                  
smsp__thread_inst_executed_pipe_tensor_op_hmma_type_hfma2_pred_on           Counter         inst            # of thread instructions executed by tensor_op_hmma_type_hfma2 where  
                                                                                                            guard predicate was true                                              
smsp__thread_inst_executed_pipe_xu_pred_on                                  Counter         inst            # of thread instructions executed by xu where guard predicate was true
smsp__thread_inst_executed_pred_off                                         Counter         inst            # of thread instructions executed where guard predicate was false     
smsp__thread_inst_executed_pred_on                                          Counter         inst            # of thread instructions executed where guard predicate was true      
smsp__thread_inst_executed_pred_on_per_inst_executed                        Ratio                           average # of active predicated-on threads per instruction executed    
smsp__threads_launched                                                      Counter         thread          # of threads launched                                                 
smsp__warp_issue_stalled_barrier_per_warp_active                            Ratio                           proportion of warps per cycle, waiting for sibling warps at a CTA     
                                                                                                            barrier                                                               
smsp__warp_issue_stalled_branch_resolving_per_warp_active                   Ratio                           proportion of warps per cycle, waiting for a branch target address to 
                                                                                                            be computed, and the warp PC to be updated                            
smsp__warp_issue_stalled_dispatch_stall_per_warp_active                     Ratio                           proportion of warps per cycle, waiting on a dispatch stall            
smsp__warp_issue_stalled_drain_per_warp_active                              Ratio                           proportion of warps per cycle, waiting after EXIT for all memory      
                                                                                                            instructions to complete so that warp resources can be freed          
smsp__warp_issue_stalled_imc_miss_per_warp_active                           Ratio                           proportion of warps per cycle, waiting for an immediate constant      
                                                                                                            cache (IMC) miss                                                      
smsp__warp_issue_stalled_lg_throttle_per_warp_active                        Ratio                           proportion of warps per cycle, waiting for a free entry in the LSU    
                                                                                                            instruction queue                                                     
smsp__warp_issue_stalled_long_scoreboard_per_warp_active                    Ratio                           proportion of warps per cycle, waiting for a scoreboard dependency on 
                                                                                                            L1TEX (local, global, surface, tex) operation                         
smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active                 Ratio                           proportion of warps per cycle, waiting for an execution pipe to be    
                                                                                                            available                                                             
smsp__warp_issue_stalled_membar_per_warp_active                             Ratio                           proportion of warps per cycle, waiting on a memory barrier            
smsp__warp_issue_stalled_mio_throttle_per_warp_active                       Ratio                           proportion of warps per cycle, waiting for a free entry in the MIO    
                                                                                                            instruction queue                                                     
smsp__warp_issue_stalled_misc_per_warp_active                               Ratio                           proportion of warps per cycle, waiting on a miscellaneous hardware    
                                                                                                            reason                                                                
smsp__warp_issue_stalled_no_instruction_per_warp_active                     Ratio                           proportion of warps per cycle, waiting to be selected for instruction 
                                                                                                            fetch, or waiting on an instruction cache miss                        
smsp__warp_issue_stalled_not_selected_per_warp_active                       Ratio                           proportion of warps per cycle, waiting for the microscheduler to      
                                                                                                            select the warp to issue                                              
smsp__warp_issue_stalled_selected_per_warp_active                           Ratio                           proportion of warps per cycle, selected by the microscheduler to      
                                                                                                            issue an instruction                                                  
smsp__warp_issue_stalled_short_scoreboard_per_warp_active                   Ratio                           proportion of warps per cycle, waiting for a scoreboard dependency on 
                                                                                                            MIO operation other than (local, global, surface, tex)                
smsp__warp_issue_stalled_sleeping_per_warp_active                           Ratio                           proportion of warps per cycle, waiting for a nanosleep to expire      
smsp__warp_issue_stalled_tex_throttle_per_warp_active                       Ratio                           proportion of warps per cycle, waiting for a free entry in the TEX    
                                                                                                            instruction queue                                                     
smsp__warp_issue_stalled_wait_per_warp_active                               Ratio                           proportion of warps per cycle, waiting on a fixed latency execution   
                                                                                                            dependency                                                            
smsp__warps_active                                                          Counter         warp            cumulative # of active warps                                          
smsp__warps_eligible                                                        Counter         warp            cumulative # of warps eligible to issue an instruction                
smsp__warps_issue_stalled_barrier                                           Counter         warp            cumulative # of warps waiting for sibling warps at a CTA barrier      
smsp__warps_issue_stalled_branch_resolving                                  Counter         warp            cumulative # of warps waiting for a branch target address to be       
                                                                                                            computed, and the warp PC to be updated                               
smsp__warps_issue_stalled_dispatch_stall                                    Counter         warp            cumulative # of warps waiting on a dispatch stall                     
smsp__warps_issue_stalled_drain                                             Counter         warp            cumulative # of warps waiting after EXIT for all memory instructions  
                                                                                                            to complete so that warp resources can be freed                       
smsp__warps_issue_stalled_imc_miss                                          Counter         warp            cumulative # of warps waiting for an immediate constant cache (IMC)   
                                                                                                            miss                                                                  
smsp__warps_issue_stalled_lg_throttle                                       Counter         warp            cumulative # of warps waiting for a free entry in the LSU instruction 
                                                                                                            queue                                                                 
smsp__warps_issue_stalled_long_scoreboard                                   Counter         warp            cumulative # of warps waiting for a scoreboard dependency on L1TEX    
                                                                                                            (local, global, surface, tex) operation                               
smsp__warps_issue_stalled_math_pipe_throttle                                Counter         warp            cumulative # of warps waiting for an execution pipe to be available   
smsp__warps_issue_stalled_membar                                            Counter         warp            cumulative # of warps waiting on a memory barrier                     
smsp__warps_issue_stalled_mio_throttle                                      Counter         warp            cumulative # of warps waiting for a free entry in the MIO instruction 
                                                                                                            queue                                                                 
smsp__warps_issue_stalled_misc                                              Counter         warp            cumulative # of warps waiting on a miscellaneous hardware reason      
smsp__warps_issue_stalled_no_instruction                                    Counter         warp            cumulative # of warps waiting to be selected for instruction fetch,   
                                                                                                            or waiting on an instruction cache miss                               
smsp__warps_issue_stalled_not_selected                                      Counter         warp            cumulative # of warps waiting for the microscheduler to select the    
                                                                                                            warp to issue                                                         
smsp__warps_issue_stalled_selected                                          Counter         warp            cumulative # of warps selected by the microscheduler to issue an      
                                                                                                            instruction                                                           
smsp__warps_issue_stalled_short_scoreboard                                  Counter         warp            cumulative # of warps waiting for a scoreboard dependency on MIO      
                                                                                                            operation other than (local, global, surface, tex)                    
smsp__warps_issue_stalled_sleeping                                          Counter         warp            cumulative # of warps waiting for a nanosleep to expire               
smsp__warps_issue_stalled_tex_throttle                                      Counter         warp            cumulative # of warps waiting for a free entry in the TEX instruction 
                                                                                                            queue                                                                 
smsp__warps_issue_stalled_wait                                              Counter         warp            cumulative # of warps waiting on a fixed latency execution dependency 
smsp__warps_launched                                                        Counter         warp            # of warps launched                                                   
smsp__warps_launched_total                                                  Counter         warp            # of warps launched including warps restored from pre-emption         
smsp__warps_restored                                                        Counter         warp            # of warp preemption-restore events                                   
sys__cycles_active                                                          Counter         cycle           # of cycles where SYS was active                                      
sys__cycles_elapsed                                                         Counter         cycle           # of cycles elapsed on SYS                                            
sys__cycles_in_frame                                                        Counter         cycle           # of cycles in user-defined frame                                     
sys__cycles_in_region                                                       Counter         cycle           # of cycles in user-defined region                                    
tpc__cycles_active                                                          Counter         cycle           # of cycles where TPC was active                                      
tpc__cycles_elapsed                                                         Counter         cycle           # of cycles elapsed on TPC                                            
tpc__cycles_in_frame                                                        Counter         cycle           # of cycles in user-defined frame                                     
tpc__cycles_in_region                                                       Counter         cycle           # of cycles in user-defined region                                    
tpc__l1tex_m_l1tex2xbar_req_cycles_active                                   Counter         cycle           # of cycles where interface from L1TEX M-Stage to XBAR was active     
tpc__l1tex_m_l1tex2xbar_throughput                                          Throughput      %               L1TEX M-Stage to XBAR throughput                                      
tpc__warp_launch_cycles_stalled_shader_cs_reason_barrier_allocation         Counter         cycle           # of cycles where CS warp launch was stalled due to barrier           
                                                                                                            allocation (non-exclusively)                                          
tpc__warp_launch_cycles_stalled_shader_cs_reason_cta_allocation             Counter         cycle           # of cycles where CS warp launch was stalled due to cta allocation    
                                                                                                            (non-exclusively)                                                     
tpc__warp_launch_cycles_stalled_shader_cs_reason_register_allocation        Counter         cycle           # of cycles where CS warp launch was stalled due to register          
                                                                                                            allocation (non-exclusively)                                          
tpc__warp_launch_cycles_stalled_shader_cs_reason_shmem_allocation           Counter         cycle           # of cycles where CS warp launch was stalled due to shmem allocation  
                                                                                                            (non-exclusively)                                                     
tpc__warps_active                                                           Counter         warp            cumulative # of warps in flight                                       
tpc__warps_active_shader_cs                                                 Counter         warp            cumulative # of active CS warps                                       
tpc__warps_active_shader_cs_realtime                                        Counter         warp            cumulative # of active CS warps                                       
==PROF== Note that these metrics must be appended with a valid suffix before profiling them. See --help for more information on --query-metrics-mode.

