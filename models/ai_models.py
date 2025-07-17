import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAIModel:
    """
    Simple AI model for resume-job matching using text analysis and similarity scoring
    """

    def __init__(self):
        self.skill_keywords = self._load_skill_keywords()
        self.industry_keywords = self._load_industry_keywords()
        self.soft_skills = self._load_soft_skills()
        self.stop_words = self._load_stop_words()

    def _load_skill_keywords(self) -> List[str]:
        """Load technical skills and keywords"""
        return [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby',
            'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',

            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'asp.net', 'laravel', 'rails',

            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
            'oracle', 'sqlite', 'dynamodb', 'cassandra', 'neo4j',

            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
            'jenkins', 'git', 'github', 'gitlab', 'ci/cd', 'devops',
            'terraform', 'ansible', 'chef', 'puppet',

            # Data Science & ML
            'machine learning', 'deep learning', 'ai', 'artificial intelligence',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'data science', 'data analysis', 'statistics', 'big data',
            'hadoop', 'spark', 'kafka', 'etl', 'data warehouse',

            # Tools & Frameworks
            'linux', 'windows', 'unix', 'bash', 'powershell', 'vim', 'vscode',
            'intellij', 'eclipse', 'jira', 'confluence', 'slack', 'teams',

            # Testing
            'testing', 'unit testing', 'integration testing', 'selenium',
            'cypress', 'jest', 'junit', 'pytest', 'tdd', 'bdd',

            # Methodologies
            'agile', 'scrum', 'kanban', 'waterfall', 'lean', 'safe',
            'project management', 'product management',

            # Business Intelligence
            'tableau', 'power bi', 'qlik', 'looker', 'excel', 'sap',
            'business intelligence', 'data visualization', 'reporting',

            # Security
            'cybersecurity', 'security', 'penetration testing', 'encryption',
            'authentication', 'authorization', 'compliance', 'gdpr',

            # Mobile
            'ios', 'android', 'react native', 'flutter', 'xamarin',
            'mobile development', 'app development'
        ]

    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load industry-specific keywords"""
        return {
            'technology': ['software', 'development', 'engineering', 'tech', 'it',
                           'programming', 'coding', 'algorithm', 'system design'],
            'finance': ['banking', 'financial', 'investment', 'trading', 'fintech',
                        'accounting', 'audit', 'compliance', 'risk management'],
            'healthcare': ['medical', 'healthcare', 'clinical', 'patient care',
                           'pharmaceutical', 'biotech', 'medical device'],
            'marketing': ['digital marketing', 'seo', 'sem', 'social media',
                          'content marketing', 'brand management', 'advertising'],
            'sales': ['sales', 'business development', 'account management',
                      'lead generation', 'customer acquisition', 'crm'],
            'consulting': ['consulting', 'strategy', 'advisory', 'transformation',
                           'change management', 'process improvement'],
            'education': ['teaching', 'education', 'training', 'curriculum',
                          'instructional design', 'e-learning'],
            'operations': ['operations', 'supply chain', 'logistics', 'procurement',
                           'vendor management', 'process optimization']
        }

    def _load_soft_skills(self) -> List[str]:
        """Load soft skills keywords"""
        return [
            'leadership', 'management', 'communication', 'teamwork', 'collaboration',
            'problem solving', 'analytical', 'creative', 'innovative', 'adaptable',
            'flexible', 'organized', 'detail-oriented', 'motivated', 'results-driven',
            'customer service', 'presentation', 'negotiation', 'conflict resolution',
            'time management', 'multitasking', 'priority management', 'mentoring',
            'coaching', 'training', 'strategic thinking', 'decision making',
            'critical thinking', 'emotional intelligence', 'interpersonal skills'
        ]

    def _load_stop_words(self) -> List[str]:
        """Load common stop words to filter out"""
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that',
            'the', 'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but',
            'they', 'have', 'had', 'what', 'said', 'each', 'which', 'their',
            'time', 'will', 'about', 'if', 'up', 'out', 'many', 'then', 'them',
            'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him',
            'has', 'two', 'more', 'very', 'after', 'words', 'long', 'than',
            'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down',
            'day', 'did', 'get', 'come', 'made', 'may', 'part'
        ]

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for analysis

        Args:
            text: Raw text

        Returns:
            List of processed words
        """
        if not text:
            return []

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)

        # Extract words (including hyphenated terms)
        words = re.findall(r'\b[\w\-]+\b', text)

        # Filter out stop words and short words
        filtered_words = [
            word for word in words
            if word not in self.stop_words and len(word) > 2
        ]

        return filtered_words

    def extract_skills_from_text(self, text: str) -> Dict[str, int]:
        """
        Extract skills from text with frequency count

        Args:
            text: Input text

        Returns:
            Dictionary of skills with their frequencies
        """
        text_lower = text.lower()
        skills_found = {}

        # Check for technical skills
        for skill in self.skill_keywords:
            count = len(re.findall(r'\b' + re.escape(skill.lower()) + r'\b', text_lower))
            if count > 0:
                skills_found[skill] = count

        # Check for soft skills
        for skill in self.soft_skills:
            count = len(re.findall(r'\b' + re.escape(skill.lower()) + r'\b', text_lower))
            if count > 0:
                skills_found[skill] = count

        return skills_found

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using cosine similarity

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        # Preprocess both texts
        words1 = self.preprocess_text(text1)
        words2 = self.preprocess_text(text2)

        if not words1 or not words2:
            return 0.0

        # Create word frequency vectors
        all_words = set(words1 + words2)

        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]

        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(a * a for a in vector1))
        magnitude2 = math.sqrt(sum(b * b for b in vector2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def calculate_skill_match(self, resume_skills: Dict[str, int],
                              job_skills: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate skill matching between resume and job

        Args:
            resume_skills: Skills found in resume
            job_skills: Skills found in job description

        Returns:
            Dictionary with skill matching analysis
        """
        # Find common skills
        common_skills = set(resume_skills.keys()) & set(job_skills.keys())

        # Find missing skills
        missing_skills = set(job_skills.keys()) - set(resume_skills.keys())

        # Calculate match percentage
        if not job_skills:
            match_percentage = 0
        else:
            match_percentage = (len(common_skills) / len(job_skills)) * 100

        # Weight skills by frequency in job description
        weighted_match = 0
        total_weight = 0

        for skill, freq in job_skills.items():
            total_weight += freq
            if skill in resume_skills:
                weighted_match += freq

        weighted_percentage = (weighted_match / total_weight * 100) if total_weight > 0 else 0

        return {
            'common_skills': list(common_skills),
            'missing_skills': list(missing_skills),
            'match_percentage': round(match_percentage, 2),
            'weighted_match_percentage': round(weighted_percentage, 2),
            'total_job_skills': len(job_skills),
            'total_resume_skills': len(resume_skills),
            'skills_overlap': len(common_skills)
        }

    def analyze_resume_sections(self, resume_sections: Dict[str, str],
                                job_description: str) -> Dict[str, float]:
        """
        Analyze how well each resume section matches the job description

        Args:
            resume_sections: Dictionary of resume sections
            job_description: Job description text

        Returns:
            Dictionary with section match scores
        """
        section_scores = {}

        for section_name, section_content in resume_sections.items():
            if section_content.strip():
                similarity = self.calculate_text_similarity(section_content, job_description)
                section_scores[section_name] = similarity

        return section_scores

    def generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate improvement recommendations based on analysis

        Args:
            analysis_result: Analysis result dictionary

        Returns:
            List of recommendations
        """
        recommendations = []

        missing_skills = analysis_result.get('missing_skills', [])
        match_percentage = analysis_result.get('match_percentage', 0)
        section_matches = analysis_result.get('section_matches', {})

        # Skill-based recommendations
        if missing_skills:
            high_priority_skills = missing_skills[:5]  # Top 5 missing skills
            recommendations.append(
                f"Consider adding these high-priority skills to your resume: {', '.join(high_priority_skills)}"
            )

        # Match percentage recommendations
        if match_percentage < 30:
            recommendations.append(
                "Your resume has low alignment with this job. Consider tailoring your resume to better match the job requirements."
            )
        elif match_percentage < 50:
            recommendations.append(
                "Your resume shows fair alignment. Focus on highlighting relevant experience and skills more prominently."
            )
        elif match_percentage < 70:
            recommendations.append(
                "Good alignment! Consider adding specific examples that demonstrate the missing skills."
            )

        # Section-based recommendations
        if section_matches:
            weak_sections = [section for section, score in section_matches.items() if score < 0.3]
            if weak_sections:
                recommendations.append(
                    f"Consider strengthening these resume sections: {', '.join(weak_sections)}"
                )

        # Experience recommendations
        experience_score = section_matches.get('experience', 0)
        if experience_score < 0.4:
            recommendations.append(
                "Add more relevant work experience examples that align with the job requirements"
            )

        # Skills section recommendations
        skills_score = section_matches.get('skills', 0)
        if skills_score < 0.5:
            recommendations.append(
                "Expand your skills section to include more relevant technical and soft skills"
            )

        # Education recommendations
        education_score = section_matches.get('education', 0)
        if education_score < 0.3:
            recommendations.append(
                "Consider highlighting relevant coursework, certifications, or educational achievements"
            )

        # Generic recommendations if no specific issues found
        if not recommendations:
            recommendations.append(
                "Your resume looks good! Consider adding quantifiable achievements and specific examples of your impact"
            )

        return recommendations[:10]  # Limit to top 10 recommendations

    def extract_keywords_with_context(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract keywords with context and frequency information

        Args:
            text: Input text

        Returns:
            Dictionary with keyword analysis
        """
        keywords = {}
        words = self.preprocess_text(text)
        word_freq = Counter(words)

        # Analyze all keywords
        all_keywords = self.skill_keywords + self.soft_skills

        for keyword in all_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in [w.lower() for w in words]:
                # Find exact matches accounting for multi-word keywords
                pattern = r'\b' + re.escape(keyword_lower) + r'\b'
                matches = re.findall(pattern, text.lower())

                if matches:
                    keywords[keyword] = {
                        'frequency': len(matches),
                        'type': 'technical' if keyword in self.skill_keywords else 'soft',
                        'importance': self._calculate_keyword_importance(keyword, text)
                    }

        return keywords

    def _calculate_keyword_importance(self, keyword: str, text: str) -> float:
        """
        Calculate the importance of a keyword based on various factors

        Args:
            keyword: The keyword to analyze
            text: Full text context

        Returns:
            Importance score between 0 and 1
        """
        text_lower = text.lower()
        keyword_lower = keyword.lower()

        # Base importance factors
        importance = 0.0

        # Frequency factor
        frequency = len(re.findall(r'\b' + re.escape(keyword_lower) + r'\b', text_lower))
        frequency_score = min(frequency * 0.2, 1.0)
        importance += frequency_score * 0.3

        # Position factor (keywords appearing early are more important)
        first_occurrence = text_lower.find(keyword_lower)
        if first_occurrence != -1:
            position_score = 1.0 - (first_occurrence / len(text_lower))
            importance += position_score * 0.2

        # Context factor (keywords in headings/sections are more important)
        context_patterns = [
            r'\n\s*' + re.escape(keyword_lower),  # At line start
            r'\*\s*' + re.escape(keyword_lower),  # After bullet point
            r':\s*' + re.escape(keyword_lower),  # After colon
        ]

        for pattern in context_patterns:
            if re.search(pattern, text_lower):
                importance += 0.1

        # Technical skills get higher base importance
        if keyword in self.skill_keywords:
            importance += 0.3

        return min(importance, 1.0)

    def analyze_resume_job_match(self, resume_text: str, resume_sections: Dict[str, str],
                                 job_description: str, job_title: str = "",
                                 job_requirements: str = "") -> Dict[str, Any]:
        """
        Comprehensive analysis of resume-job match

        Args:
            resume_text: Full resume text
            resume_sections: Dictionary of resume sections
            job_description: Job description text
            job_title: Job title (optional)
            job_requirements: Specific job requirements (optional)

        Returns:
            Complete analysis result
        """
        try:
            logger.info("Starting comprehensive resume-job analysis")

            # Combine job description and requirements
            full_job_text = job_description
            if job_requirements:
                full_job_text += "\n\n" + job_requirements

            # Extract skills from both texts
            resume_skills = self.extract_skills_from_text(resume_text)
            job_skills = self.extract_skills_from_text(full_job_text)

            # Calculate skill matching
            skill_match = self.calculate_skill_match(resume_skills, job_skills)

            # Analyze resume sections
            section_matches = self.analyze_resume_sections(resume_sections, full_job_text)

            # Calculate overall text similarity
            overall_similarity = self.calculate_text_similarity(resume_text, full_job_text)

            # Calculate weighted match score
            skill_weight = 0.4
            section_weight = 0.3
            similarity_weight = 0.3

            weighted_score = (
                    skill_match['weighted_match_percentage'] * skill_weight +
                    (sum(section_matches.values()) / len(
                        section_matches) * 100 if section_matches else 0) * section_weight +
                    overall_similarity * 100 * similarity_weight
            )

            # Extract keywords with context
            resume_keywords = self.extract_keywords_with_context(resume_text)
            job_keywords = self.extract_keywords_with_context(full_job_text)

            # Find common and missing keywords
            common_keywords = {}
            missing_keywords = {}

            for keyword, data in job_keywords.items():
                if keyword in resume_keywords:
                    common_keywords[keyword] = {
                        'job_freq': data['frequency'],
                        'resume_freq': resume_keywords[keyword]['frequency'],
                        'importance': data['importance']
                    }
                else:
                    missing_keywords[keyword] = data['frequency']

            # Determine match level
            if weighted_score >= 80:
                match_level = "Excellent"
            elif weighted_score >= 60:
                match_level = "Good"
            elif weighted_score >= 40:
                match_level = "Fair"
            elif weighted_score >= 20:
                match_level = "Poor"
            else:
                match_level = "Very Poor"

            # Compile complete analysis
            analysis_result = {
                'match_percentage': round(weighted_score, 2),
                'match_level': match_level,
                'skill_match_percentage': skill_match['weighted_match_percentage'],
                'text_similarity': round(overall_similarity * 100, 2),
                'common_skills': skill_match['common_skills'],
                'missing_skills': skill_match['missing_skills'],
                'section_matches': section_matches,
                'common_keywords': common_keywords,
                'missing_keywords': missing_keywords,
                'total_skills_found': len(resume_skills),
                'total_job_skills': len(job_skills),
                'skill_overlap': skill_match['skills_overlap']
            }

            # Generate recommendations
            analysis_result['recommendations'] = self.generate_recommendations(analysis_result)

            logger.info(f"Analysis complete. Match score: {weighted_score:.2f}%")

            return analysis_result

        except Exception as e:
            logger.error(f"Error in resume-job analysis: {str(e)}")
            return {
                'match_percentage': 0,
                'match_level': 'Error',
                'error': str(e),
                'recommendations': ['Unable to analyze due to an error. Please try again.']
            }

    def get_industry_analysis(self, text: str) -> Dict[str, float]:
        """
        Analyze which industries the text best matches

        Args:
            text: Input text

        Returns:
            Dictionary with industry match scores
        """
        industry_scores = {}
        text_lower = text.lower()

        for industry, keywords in self.industry_keywords.items():
            score = 0
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
                score += count

            # Normalize score
            if keywords:
                industry_scores[industry] = score / len(keywords)

        return industry_scores

    def extract_years_experience(self, text: str) -> Optional[int]:
        """
        Extract years of experience from text

        Args:
            text: Input text

        Returns:
            Years of experience or None if not found
        """
        patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(\d+)\+\s*(?:years?|yrs?)',
            r'over\s*(\d+)\s*(?:years?|yrs?)',
            r'more\s*than\s*(\d+)\s*(?:years?|yrs?)'
        ]

        text_lower = text.lower()
        max_years = 0

        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue

        return max_years if max_years > 0 else None

    def calculate_resume_score(self, resume_text: str, resume_sections: Dict[str, str]) -> Dict[str, Any]:
        """
        Calculate overall resume quality score

        Args:
            resume_text: Full resume text
            resume_sections: Dictionary of resume sections

        Returns:
            Resume quality analysis
        """
        score = 0
        max_score = 100
        feedback = []

        # Check for essential sections
        essential_sections = ['contact', 'experience', 'skills', 'education']
        sections_present = [section for section in essential_sections
                            if resume_sections.get(section, '').strip()]

        section_score = (len(sections_present) / len(essential_sections)) * 30
        score += section_score

        if len(sections_present) < len(essential_sections):
            missing = set(essential_sections) - set(sections_present)
            feedback.append(f"Missing sections: {', '.join(missing)}")

        # Check word count
        word_count = len(resume_text.split())
        if 200 <= word_count <= 800:
            score += 20
        elif word_count < 200:
            feedback.append("Resume is too short. Consider adding more details.")
        else:
            feedback.append("Resume is too long. Consider being more concise.")

        # Check for skills
        skills_found = self.extract_skills_from_text(resume_text)
        skills_score = min(len(skills_found) * 2, 25)
        score += skills_score

        if len(skills_found) < 5:
            feedback.append("Consider adding more relevant skills.")

        # Check for quantifiable achievements
        achievement_patterns = [
            r'\d+%', r'\$\d+', r'\d+\s*(?:million|thousand|k)',
            r'increased?.*by.*\d+', r'reduced?.*by.*\d+',
            r'improved?.*by.*\d+', r'saved?.*\$?\d+'
        ]

        achievement_count = 0
        for pattern in achievement_patterns:
            achievement_count += len(re.findall(pattern, resume_text, re.IGNORECASE))

        achievement_score = min(achievement_count * 5, 25)
        score += achievement_score

        if achievement_count < 3:
            feedback.append("Add more quantifiable achievements and results.")

        return {
            'score': round(score, 2),
            'max_score': max_score,
            'percentage': round((score / max_score) * 100, 2),
            'feedback': feedback,
            'sections_present': sections_present,
            'word_count': word_count,
            'skills_count': len(skills_found),
            'achievements_count': achievement_count
        }