#ifndef SNDE_CPP_CALIBRATION_FUNCTION_HPP

#include "snde/recmath.hpp"
#include "snde/snde_error.hpp"
#include "snde/recmath_cppfunction.hpp"

#include <opencv2/opencv.hpp>



namespace snde2_fn_ex {

    
    template<typename T>
    cv::Mat copyparamstocvmat(std::shared_ptr<snde::multi_ndarray_recording> rec, snde_index arr) {
        if(rec->layouts.at(arr).dimlen.size() == 2) {
            size_t w = rec->layouts.at(arr).dimlen[0];
            size_t h = rec->layouts.at(arr).dimlen[1];

            int type = 0;
            if (rec->ndinfo(arr)->typenum == SNDE_RTN_FLOAT32) {
                type = CV_32F;
            }
            else if (rec->ndinfo(arr)->typenum == SNDE_RTN_UINT16) {
                type = CV_16U;
            }
            else if (rec->ndinfo(arr)->typenum == SNDE_RTN_UINT8) {
                type = CV_8U;
            }
            else {
                throw snde::snde_error("calibration_function_cpp::copyparamstocvmat -- Unknown type");
            }


            cv::Mat retval(w, h, type);

            for (size_t i = 0; i < w; i++) {
                for (size_t j = 0; j < h; j++) {
                    retval.at<T>(i, j) = *(T*)rec->element_dataptr(arr, { i,j });
                }
            }

            return retval;

        }
        else {
            throw snde::snde_error("calibration_function_cpp::copyparamstocvmat -- Only works on 2D recordings");
        }

      
    }

    template<typename T>
    cv::Mat copyparamstocvmat(std::shared_ptr<snde::multi_ndarray_recording> rec, std::string arrname) {
          snde_index arr = rec->name_mapping.at(arrname);
          return copyparamstocvmat<T>(rec, arr);
    }



  
  template <typename T>
  class camera_calibration: public snde::recmath_cppfuncexec<std::shared_ptr<snde::multi_ndarray_recording>,std::shared_ptr<snde::multi_ndarray_recording>,snde_bool>
  {
  public:
      camera_calibration(std::shared_ptr<snde::recording_set_state> rss,std::shared_ptr<snde::instantiated_math_function> inst) :
      snde::recmath_cppfuncexec<std::shared_ptr<snde::multi_ndarray_recording>,std::shared_ptr<snde::multi_ndarray_recording>,snde_bool>(rss,inst)
    {
      
    }
    
    // These typedefs are regrettably necessary and will need to be updated according to the parameter signature of your function
    // https://stackoverflow.com/questions/1120833/derived-template-class-access-to-base-class-member-data
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<snde::multi_ndarray_recording>,std::shared_ptr<snde::multi_ndarray_recording>,snde_bool>::metadata_function_override_type metadata_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<snde::multi_ndarray_recording>,std::shared_ptr<snde::multi_ndarray_recording>,snde_bool>::lock_alloc_function_override_type lock_alloc_function_override_type;
    typedef typename snde::recmath_cppfuncexec<std::shared_ptr<snde::multi_ndarray_recording>,std::shared_ptr<snde::multi_ndarray_recording>,snde_bool>::exec_function_override_type exec_function_override_type;

    // just using the default for decide_new_revision and compute_options
    
    std::shared_ptr<metadata_function_override_type> define_recs(std::shared_ptr<snde::multi_ndarray_recording> recording, std::shared_ptr<snde::multi_ndarray_recording> calibrec, snde_bool crop)
    {
      // define_recs code
      snde::snde_debug(SNDE_DC_APP,"define_recs()");
      // Use of "this" in the next line for the same reason as the typedefs, above
      std::shared_ptr<snde::multi_ndarray_recording> result_rec = snde::create_recording_math<snde::multi_ndarray_recording>(this->get_result_channel_path(0),this->rss,1);
      result_rec->define_array(0, recording->ndinfo(0)->typenum, "calibimg");
      // ***!!! Should provide means to set allocation manager !!!***
      
      return std::make_shared<metadata_function_override_type>([ this,result_rec,recording,calibrec,crop]() {
	// metadata code
	snde::snde_debug(SNDE_DC_APP,"metadata()");
	

    std::shared_ptr<snde::constructible_metadata> metadata;
    result_rec->metadata = snde::MergeMetadata(metadata, recording->metadata);
	result_rec->mark_metadata_done();
	
	return std::make_shared<lock_alloc_function_override_type>([ this,result_rec,recording,calibrec,crop]() {
	  // lock_alloc code
	  
	  result_rec->allocate_storage(0, recording->layouts.at(0).dimlen);

	  // locking is only required for certain recordings
	  // with special storage under certain conditions,
	  // however it is always good to explicitly request
	  // the locks, as the locking is a no-op if
	  // locking is not actually required.
    std::vector<std::pair<std::shared_ptr<snde::multi_ndarray_recording>, std::pair<size_t, bool>>> recrefs_to_lock = {
					  { result_rec, { 0, true } }, // colorimage
					};

    for (snde_index arraynum = 0; arraynum < recording->mndinfo()->num_arrays; arraynum++) {
						recrefs_to_lock.emplace_back(std::make_pair(recording, std::make_pair(arraynum, false)));
					}

    for (snde_index arraynum = 0; arraynum < calibrec->mndinfo()->num_arrays; arraynum++) {
						recrefs_to_lock.emplace_back(std::make_pair(calibrec, std::make_pair(arraynum, false)));
					}

    snde::rwlock_token_set locktokens = this->lockmgr->lock_recording_arrays(recrefs_to_lock, false);
    
	  

	  
	  return std::make_shared<exec_function_override_type>([ this,locktokens,result_rec,recording,calibrec,crop]() {
	    // exec code
	    
      cv::Mat mtx = copyparamstocvmat<snde_float32>(calibrec, "cam_mtx");
      cv::Vec<snde_float32, 14> dist((snde_float32*)calibrec->void_shifted_arrayptr("cam_dist"));

      cv::Mat newmtx = copyparamstocvmat<snde_float32>(calibrec, "cam_newmtx");
      cv::Mat cvin = copyparamstocvmat<T>(recording, 0);
      
      cv::Mat cvout;

      size_t w = result_rec->layouts.at(0).dimlen[0];
      size_t h = result_rec->layouts.at(0).dimlen[1];
      
      cv::undistort(cvin, cvout, mtx, dist, newmtx);

      for (size_t i=0; i < w; i++) {
        for (size_t j=0; j< h; j++) {
          *(T*)result_rec->element_dataptr(0, {i,j}) = cvout.at<T>(i,j);
        }
      }
      
      
      



	    
	    snde::unlock_rwlock_token_set(locktokens); // lock must be released prior to mark_data_ready() 

	    result_rec->mark_data_ready();
	  }); 
	});
      });
    }
    
    
    
  };


  
  std::shared_ptr<snde::math_function> define_calibration_function()
  {
      std::shared_ptr<snde::math_function> newfunc = std::make_shared<snde::cpp_math_function>([](std::shared_ptr<snde::recording_set_state> rss, std::shared_ptr<snde::instantiated_math_function> inst) {
          std::shared_ptr<snde::executing_math_function> executing;

          executing = snde::make_cppfuncexec_integertypes<camera_calibration>(rss, inst);
          if (!executing) {
              executing = snde::make_cppfuncexec_floatingtypes<camera_calibration>(rss, inst);
          }

          if (!executing) {
              throw snde::snde_error("In attempting to call math function %s, first parameter has unsupported data type.", inst->definition->definition_command.c_str());
          }
          return executing;
          });
      //newfunc->self_dependent=true;
      newfunc->new_revision_optional = true;
      return newfunc;
    
  }


  static std::shared_ptr<snde::math_function> calibration_function=define_calibration_function();

  // Register the math function into the C++ database
  // This should use the same python-accessible name for
  // maximum interoperability

  static int registered_calibration_function = register_math_function("spatialnde2_cpp_calibration_function.calibration_function",calibration_function);

};

#endif // SNDE_CPP_CALIBRATION_FUNCTION_HPP
