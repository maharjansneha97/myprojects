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
        self.risk_keywords = {
            'high': ['fire', 'weapon', 'knife', 'gun', 'smoke'],
            'medium': ['car', 'motorcycle', 'truck', 'traffic', 'stairs'],
            'low': ['person', 'chair', 'book', 'laptop']
        }
        
        self.activity_inference = {
            'office': ['working', 'meetings', 'desk work', 'computing'],
            'kitchen': ['cooking', 'food preparation', 'dining', 'cleaning'],
            'street': ['transportation', 'commuting', 'walking', 'traffic management'],
            'classroom': ['learning', 'teaching', 'studying', 'lecturing'],
            'gym': ['exercise', 'fitness', 'training', 'workout'],
            'park': ['recreation', 'relaxation', 'outdoor activities', 'picnic'],
            'beach': ['swimming', 'sunbathing', 'relaxation', 'water sports'],
            'forest': ['hiking', 'nature exploration', 'photography', 'camping'],
            'mountain': ['climbing', 'hiking', 'sightseeing', 'adventuring'],
            'living_room': ['relaxing', 'socializing', 'watching TV', 'reading']
        }
    
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
        
        if scene_type == 'indoor':
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
        risk_factors = []
        
        for det in detections:
            obj_name = det['class_name']
            
            # Check high risk objects
            if any(keyword in obj_name.lower() 
                   for keyword in self.risk_keywords['high']):
                risk_score += 3
                risk_factors.append(f"Potential hazard: {obj_name}")
            
            # Check medium risk objects
            elif any(keyword in obj_name.lower() 
                     for keyword in self.risk_keywords['medium']):
                risk_score += 1
                risk_factors.append(f"Moderate concern: {obj_name}")
        
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
            'factors': risk_factors
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
        
        if 'laptop' in object_names or 'keyboard' in object_names:
            activities.append('computer work')
        
        if 'book' in object_names:
            activities.append('reading')
        
        if any(v in object_names for v in ['car', 'truck', 'bus']):
            activities.append('transportation')
        
        if 'sports ball' in object_names or 'tennis racket' in object_names:
            activities.append('sports or recreation')
        
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
        
        if any(d['class_name'] in ['car', 'truck', 'bus'] for d in detections):
            use_cases.append('Vehicle tracking and management')
        
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