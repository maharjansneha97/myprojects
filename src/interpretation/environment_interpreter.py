"""
Environment Interpretation Module
Generates intelligent natural language descriptions combining object detection
and scene classification results using rule-based reasoning
"""

import json
from datetime import datetime


class EnvironmentInterpreter:
    """
    Rule-based interpretation system that combines:
    - Object detection results
    - Scene classification results
    - Contextual reasoning
    
    Generates human-readable environment descriptions
    """
    
    def __init__(self):
        """Initialize interpreter with reasoning rules"""
        # Contextual Probabilities: What objects are UNLIKELY in which scenes
        # Objects not in the 'probable' list will be suppressed if confidence < threshold
        self.contextual_sane_checks = {
            'Airport': ['aeroplane', 'bus', 'car', 'person', 'train'],
            'Beach': ['boat', 'person', 'bird'],
            'Commercial': ['car', 'bus', 'person', 'bicycle', 'motorbike'],
            'DenseResidential': ['car', 'person', 'pottedplant', 'bicycle', 'motorbike'],
            'Farmland': ['cow', 'sheep', 'horse', 'bird', 'person'],
            'Forest': ['bird', 'cat', 'dog', 'person'],
            'Industrial': ['car', 'bus', 'train', 'person'],
            'Meadow': ['cow', 'sheep', 'horse', 'bird', 'person'],
            'Mountain': ['bird', 'horse', 'aeroplane', 'person'],
            'Park': ['person', 'dog', 'bicycle', 'bird'],
            'Parking': ['car', 'bus', 'bicycle', 'person'],
            'Port': ['boat', 'car', 'bus', 'person'],
            'RailwayStation': ['train', 'person', 'bus'],
            'River': ['boat', 'bird', 'person'],
            'Street': ['car', 'bus', 'person', 'bicycle', 'motorbike'],
            'Stadium': ['person'],
            'Viaduct': ['train', 'car', 'bus', 'person'],
            'MediumResidential': ['car', 'person', 'pottedplant', 'bicycle'],
            'SparseResidential': ['car', 'person', 'pottedplant', 'horse', 'cow'],
            'School': ['person', 'bus', 'car', 'bicycle'],
            'Center': ['person', 'car', 'bus', 'bicycle']
        }
        self.risk_keywords = {
            'high': ['fire', 'weapon', 'procession'],
            'medium': ['car', 'bus', 'motorbike', 'train', 'aeroplane', 'boat'],
            'low': ['person', 'chair', 'bottle', 'sofa', 'diningtable', 'tvmonitor', 'pottedplant', 'bicycle', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep']
        }
        
        self.activity_inference = {
            'Airport': ['air travel', 'logistics', 'transportation'],
            'BareLand': ['land development', 'geographical observation'],
            'BaseballField': ['sports', 'recreation'],
            'Beach': ['recreation', 'tourism'],
            'Bridge': ['transportation', 'infrastructure monitoring'],
            'Center': ['urban activity', 'commerce'],
            'Church': ['religious activities', 'community gathering'],
            'Commercial': ['shopping', 'business transactions'],
            'DenseResidential': ['urban living', 'housing'],
            'Desert': ['environmental monitoring', 'exploration'],
            'Farmland': ['agriculture', 'food production'],
            'Forest': ['nature conservation', 'forestry'],
            'Industrial': ['manufacturing', 'industrial production'],
            'Meadow': ['grazing', 'nature observation'],
            'MediumResidential': ['suburban living', 'housing'],
            'Mountain': ['scenic viewing', 'hiking'],
            'Park': ['recreation', 'leisure'],
            'Parking': ['vehicle storage', 'urban logistics'],
            'Playground': ['recreation', 'education'],
            'Pond': ['ecology monitoring', 'recreation'],
            'Port': ['shipping', 'logistics', 'trade'],
            'RailwayStation': ['public transport', 'commuting'],
            'Resort': ['tourism', 'vacationing'],
            'River': ['water management', 'transportation'],
            'School': ['education', 'learning'],
            'SparseResidential': ['rural living', 'housing'],
            'Square': ['public gathering', 'city life'],
            'Stadium': ['sports events', 'large gatherings'],
            'StorageTanks': ['resource storage', 'industrial logistics'],
            'Viaduct': ['transportation', 'infrastructure']
        }
        
    def _filter_detections_by_context(self, detections, scene_info):
        """
        Filter detections that are highly unlikely in the given scene
        """
        primary_scene = scene_info.get('primary_scene', '').lower()
        if primary_scene not in self.contextual_sane_checks:
            return detections
            
        probable_objects = self.contextual_sane_checks[primary_scene]
        filtered = []
        
        for det in detections:
            obj_name = det['class_name'].lower()
            confidence = det['confidence']
            
            # If object is not probable in this scene, it needs moderately high confidence to be kept
            if obj_name not in probable_objects:
                # Require 0.6 confidence for "out of context" objects (lowered from 0.85)
                if confidence > 0.6:
                    filtered.append(det)
                else:
                    print(f"DEBUG: Suppressing '{obj_name}' in '{primary_scene}' (conf: {confidence})")
            else:
                filtered.append(det)
                
        return filtered
    
    def interpret_environment(self, detection_results, classification_results, 
                            detailed=True):
        """
        Generate comprehensive environment interpretation
        
        Args:
            detection_results: Dict from object detector
            classification_results: Dict from scene classifier
            detailed: Whether to generate detailed interpretation
            
        Returns:
            dict: Complete interpretation with description and metadata
        """
        detections = detection_results.get('detections', [])
        scene_info = classification_results
        
        # 0. Apply Contextual Sanity Filter
        detections = self._filter_detections_by_context(detections, scene_info)
        
        # 0.5 Semantic Cross-Validation (Mismatched Scene Fix)
        # If we see indoor/ground objects but the scene is aerial, override it.
        ground_indicators = ['sofa', 'tvmonitor', 'chair', 'diningtable', 'bottle', 'pottedplant', 'cat', 'dog', 'bicycle', 'motorbike']
        aerial_scenes = ['Airport', 'Stadium', 'Port', 'Viaduct', 'StorageTanks', 'Industrial', 'RailwayStation']
        
        detected_names = [d['class_name'].lower() for d in detections]
        has_ground_objects = any(name in ground_indicators for name in detected_names)
        
        if has_ground_objects and scene_info.get('primary_scene') in aerial_scenes:
            # Override with a generic indoor/ground label
            print(f"DEBUG: Semantic Override! Scene was '{scene_info['primary_scene']}' but saw ground objects {detected_names}")
            scene_info['primary_scene'] = 'Indoor/Home'
            scene_info['scene_type'] = 'indoor'
            scene_info['confidence'] = 0.9 # Force high confidence for the corrected label
        
        # Generate interpretation components
        scene_description = self._describe_scene(scene_info)
        object_description = self._describe_objects(detections)
        context_description = self._describe_context(detections, scene_info)
        risk_assessment = self._assess_risk(detections)
        activity_inference = self._infer_activities(detections, scene_info)
        use_cases = self._suggest_use_cases(detections, scene_info)
        
        # Combine into natural language
        if detailed:
            interpretation = self._generate_detailed_description(
                scene_description,
                object_description,
                context_description,
                activity_inference,
                risk_assessment
            )
        else:
            interpretation = self._generate_brief_description(
                scene_description,
                object_description
            )
        
        # Compile results
        results = {
            'interpretation': interpretation,
            'scene_type': scene_info.get('scene_type', 'unknown'),
            'primary_scene': scene_info.get('primary_scene', 'unknown'),
            'object_count': detection_results.get('detection_count', 0),
            'risk_level': risk_assessment['level'],
            'suggested_activities': activity_inference,
            'use_cases': use_cases,
            'confidence_score': self._calculate_confidence(
                detection_results, classification_results
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _describe_scene(self, scene_info):
        """
        Generate scene description
        
        Args:
            scene_info: Scene classification results
            
        Returns:
            str: Scene description
        """
        scene_type = scene_info.get('scene_type', 'unknown')
        primary_scene = scene_info.get('primary_scene', 'environment')
        confidence = scene_info.get('confidence', 0)
        
        # If confidence is very low, treat as generic ground environment
        # to avoid forcing an aerial category on a standard photo
        if confidence < 0.4:
            description = "a generic environment"
        elif scene_type == 'indoor':
            description = f"an indoor {primary_scene} environment"
        elif scene_type == 'outdoor':
            description = f"an outdoor {primary_scene} setting"
        else:
            description = f"a {primary_scene} environment"
        
        return description
    
    def _describe_objects(self, detections):
        """
        Generate object description
        
        Args:
            detections: List of detected objects
            
        Returns:
            str: Object description
        """
        if not detections:
            return "no specific objects detected"
        
        # Count objects by type
        object_counts = {}
        for det in detections:
            name = det['class_name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Generate description
        if len(object_counts) == 0:
            return "no objects"
        elif len(object_counts) == 1:
            obj, count = list(object_counts.items())[0]
            if count == 1:
                return f"one {obj}"
            else:
                return f"{count} {obj}s"
        elif len(object_counts) == 2:
            items = list(object_counts.items())
            obj1, count1 = items[0]
            obj2, count2 = items[1]
            
            part1 = f"{count1} {obj1}" + ("s" if count1 > 1 else "")
            part2 = f"{count2} {obj2}" + ("s" if count2 > 1 else "")
            
            return f"{part1} and {part2}"
        else:
            # Multiple objects
            items = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
            top_objects = items[:3]
            
            descriptions = []
            for obj, count in top_objects:
                if count == 1:
                    descriptions.append(f"a {obj}")
                else:
                    descriptions.append(f"{count} {obj}s")
            
            if len(items) > 3:
                return ", ".join(descriptions) + ", and other objects"
            else:
                return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
    
    def _describe_context(self, detections, scene_info):
        """
        Generate contextual description
        
        Args:
            detections: List of detected objects
            scene_info: Scene classification results
            
        Returns:
            str: Context description
        """
        people_count = sum(1 for d in detections if d['class_name'] == 'person')
        vehicle_count = sum(1 for d in detections 
                           if d['class_name'] in ['car', 'truck', 'bus', 'motorcycle'])
        
        context_parts = []
        
        if people_count > 0:
            if people_count == 1:
                context_parts.append("with one person present")
            else:
                context_parts.append(f"with {people_count} people present")
        
        if vehicle_count > 0:
            if vehicle_count == 1:
                context_parts.append("a vehicle is visible")
            else:
                context_parts.append(f"{vehicle_count} vehicles are visible")
        
        animal_count = sum(1 for d in detections if d['class_name'] == 'animal')
        if animal_count > 0:
            if animal_count == 1:
                context_parts.append("an animal is present")
            else:
                context_parts.append(f"{animal_count} animals are present")
        
        if not context_parts:
            return ""
        
        return ", ".join(context_parts)
    
    def _assess_risk(self, detections):
        """
        Assess potential risk level in environment
        
        Args:
            detections: List of detected objects
            
        Returns:
            dict: Risk assessment
        """
        risk_score = 0
        risk_factors = set()
        
        # Object counts for density assessment
        counts = {}
        for det in detections:
            name = det['class_name']
            counts[name] = counts.get(name, 0) + 1
            
        # 1. High Density Risk (Traffic/Crowds)
        if counts.get('car', 0) > 10 or counts.get('bus', 0) > 3:
            risk_score += 2
            risk_factors.add("High traffic density")
            
        if counts.get('person', 0) > 15:
            risk_score += 2
            risk_factors.add("High crowd density")
            
        # 2. Specific Danger Scenarios (Proximity)
        if 'aeroplane' in counts and 'bird' in counts:
            risk_score += 3
            risk_factors.add("Wildlife strike risk (Birds near aircraft)")
            
        # 3. Keyword based risks
        for det in detections:
            obj_name = det['class_name'].lower()
            if any(k in obj_name for k in self.risk_keywords['high']):
                risk_score += 3
                risk_factors.add(f"Critical hazard: {det['class_name']}")
            elif any(k in obj_name for k in self.risk_keywords['medium']):
                risk_score += 1
                risk_factors.add(f"Moderate concern: {det['class_name']}")
        
        # Determine risk level
        if risk_score >= 3:
            level = 'high'
        elif risk_score >= 1:
            level = 'medium'
        else:
            level = 'low'
        
        return {
            'level': level,
            'score': risk_score,
            'factors': sorted(list(risk_factors))
        }
    
    def _infer_activities(self, detections, scene_info):
        """
        Infer likely activities in the environment
        
        Args:
            detections: List of detected objects
            scene_info: Scene classification results
            
        Returns:
            list: Likely activities
        """
        scene_type = scene_info.get('primary_scene', '')
        activities = []
        
        # Scene-based activities
        if scene_type in self.activity_inference:
            activities.extend(self.activity_inference[scene_type])
        
        # Object-based activities
        object_names = [d['class_name'] for d in detections]
        
        if 'tvmonitor' in object_names:
            activities.append('watching screen')
            if scene_type == 'office':
                activities.append('computer work')
        
        if 'diningtable' in object_names:
            activities.append('dining')
        
        if any(v in object_names for v in ['car', 'bus', 'train', 'motorbike']):
            activities.append('transportation')
            
        if 'bicycle' in object_names:
            activities.append('cycling')
        
        if any(a in object_names for a in ['cat', 'dog', 'horse']):
            activities.append('interacting with animals')
            
        if any(a in object_names for a in ['cow', 'sheep']):
            activities.append('grazing/farming')
            
        if 'boat' in object_names:
            activities.append('boating/water travel')
            
        if 'aeroplane' in object_names:
            activities.append('air travel')
        
        # Remove duplicates
        activities = list(set(activities))
        
        return activities[:5]  # Top 5 activities
    
    def _suggest_use_cases(self, detections, scene_info):
        """
        Suggest potential use cases for the environment
        
        Args:
            detections: List of detected objects
            scene_info: Scene classification results
            
        Returns:
            list: Suggested use cases
        """
        use_cases = []
        scene_type = scene_info.get('scene_type', '')
        
        if scene_type == 'indoor':
            use_cases.extend([
                'Smart home automation',
                'Security monitoring',
                'Space utilization analysis'
            ])
        elif scene_type == 'outdoor':
            use_cases.extend([
                'Traffic monitoring',
                'Public safety surveillance',
                'Urban planning analysis'
            ])
        
        # Object-specific use cases
        person_count = sum(1 for d in detections if d['class_name'] == 'person')
        
        if person_count > 5:
            use_cases.append('Crowd density monitoring')
        
        if any(d['class_name'] in ['car', 'bus', 'motorbike', 'train'] for d in detections):
            use_cases.append('Vehicle tracking and management')

        if any(d['class_name'] in ['cow', 'sheep', 'horse'] for d in detections):
            use_cases.append('Livestock monitoring')
            
        if any(d['class_name'] in ['aeroplane', 'boat', 'train'] for d in detections):
            use_cases.append('Transport logistics')
        
        return use_cases[:4]  # Top 4 use cases
    
    def _calculate_confidence(self, detection_results, classification_results):
        """
        Calculate overall confidence score
        
        Args:
            detection_results: Detection results
            classification_results: Classification results
            
        Returns:
            float: Overall confidence (0-1)
        """
        det_confidence = detection_results.get('detections', [])
        if det_confidence:
            avg_det_conf = sum(d['confidence'] for d in det_confidence) / len(det_confidence)
        else:
            avg_det_conf = 0.5
        
        class_confidence = classification_results.get('confidence', 0.5)
        
        # Weighted average
        overall = (avg_det_conf * 0.6) + (class_confidence * 0.4)
        
        return round(overall, 3)
    
    def _generate_detailed_description(self, scene_desc, object_desc, 
                                      context_desc, activities, risk):
        """
        Generate detailed interpretation
        
        Returns:
            str: Detailed description
        """
        parts = [
            f"This image represents {scene_desc} containing {object_desc}."
        ]
        
        if context_desc:
            parts.append(f"The scene shows {context_desc}.")
        
        if activities:
            activity_str = ", ".join(activities[:3])
            parts.append(f"Typical activities include: {activity_str}.")
        
        if risk['level'] == 'high':
            parts.append(f"⚠️ Risk Level: HIGH. {', '.join(risk['factors'])}")
        elif risk['level'] == 'medium':
            parts.append(f"⚠️ Risk Level: MEDIUM. Exercise caution.")
        else:
            parts.append("✓ Risk Level: LOW. Safe environment.")
        
        return " ".join(parts)
    
    def _generate_brief_description(self, scene_desc, object_desc):
        """
        Generate brief interpretation
        
        Returns:
            str: Brief description
        """
        return f"This is {scene_desc} with {object_desc}."


# Example usage
if __name__ == "__main__":
    print("Environment Interpretation Module")
    print("=" * 60)
    print("Features:")
    print("• Rule-based reasoning combining ML outputs")
    print("• Natural language generation")
    print("• Risk assessment")
    print("• Activity inference")
    print("• Use case suggestions")
    print("=" * 60)