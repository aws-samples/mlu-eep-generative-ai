import json

from random import randint
from typing import Dict, Any


class Quiz:
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the component.
        """
        self.name = "Quiz"
        self.iife_script = self.iife_script = """<script>var Quiz=function(){"use strict";var M=document.createElement("style");M.textContent=`.quiz-wrapper.svelte-fk6ar3{padding:1rem}.footer.svelte-fk6ar3{display:flex;align-items:center}h2.svelte-fk6ar3{font-size:1.5rem;margin-bottom:2rem;color:#232f3e}p.svelte-fk6ar3{font-size:16px}.options.svelte-fk6ar3{display:grid;grid-template-columns:repeat(2,50%);grid-template-rows:repeat(2,1fr);width:100%;margin:auto;justify-content:center}.mlu-quizquestion-option-button.svelte-fk6ar3{padding:1rem;margin:.5rem}.submit-button.svelte-fk6ar3{padding:1rem;margin:.5rem;width:90px;color:#fff;background-color:coral}.active.svelte-fk6ar3{background-color:#232f3e;color:#fff}.correct-answer.svelte-fk6ar3{background-color:green}.incorrect-answer.svelte-fk6ar3{background-color:red;text-decoration:line-through}.available.svelte-fk6ar3{pointer-events:none;opacity:.6}
`,document.head.appendChild(M);function I(){}function P(e){return e()}function W(){return Object.create(null)}function O(e){e.forEach(P)}function X(e){return typeof e=="function"}function Z(e,t){return e!=e?t==t:e!==t||e&&typeof e=="object"||typeof e=="function"}function x(e){return Object.keys(e).length===0}function m(e,t){e.appendChild(t)}function j(e,t,n){e.insertBefore(t,n||null)}function z(e){e.parentNode&&e.parentNode.removeChild(e)}function ee(e,t){for(let n=0;n<e.length;n+=1)e[n]&&e[n].d(t)}function k(e){return document.createElement(e)}function A(e){return document.createTextNode(e)}function S(){return A(" ")}function te(){return A("")}function Y(e,t,n,r){return e.addEventListener(t,n,r),()=>e.removeEventListener(t,n,r)}function v(e,t,n){n==null?e.removeAttribute(t):e.getAttribute(t)!==n&&e.setAttribute(t,n)}function ne(e){return Array.from(e.childNodes)}function T(e,t){t=""+t,e.wholeText!==t&&(e.data=t)}function p(e,t,n){e.classList[n?"add":"remove"](t)}let L;function N(e){L=e}const $=[],D=[],Q=[],F=[],re=Promise.resolve();let B=!1;function oe(){B||(B=!0,re.then(H))}function R(e){Q.push(e)}const G=new Set;let E=0;function H(){if(E!==0)return;const e=L;do{try{for(;E<$.length;){const t=$[E];E++,N(t),le(t.$$)}}catch(t){throw $.length=0,E=0,t}for(N(null),$.length=0,E=0;D.length;)D.pop()();for(let t=0;t<Q.length;t+=1){const n=Q[t];G.has(n)||(G.add(n),n())}Q.length=0}while($.length);for(;F.length;)F.pop()();B=!1,G.clear(),N(e)}function le(e){if(e.fragment!==null){e.update(),O(e.before_update);const t=e.dirty;e.dirty=[-1],e.fragment&&e.fragment.p(e.ctx,t),e.after_update.forEach(R)}}const ie=new Set;function se(e,t){e&&e.i&&(ie.delete(e),e.i(t))}function ce(e,t,n,r){const{fragment:l,after_update:i}=e.$$;l&&l.m(t,n),r||R(()=>{const s=e.$$.on_mount.map(P).filter(X);e.$$.on_destroy?e.$$.on_destroy.push(...s):O(s),e.$$.on_mount=[]}),i.forEach(R)}function fe(e,t){const n=e.$$;n.fragment!==null&&(O(n.on_destroy),n.fragment&&n.fragment.d(t),n.on_destroy=n.fragment=null,n.ctx=[])}function ue(e,t){e.$$.dirty[0]===-1&&($.push(e),oe(),e.$$.dirty.fill(0)),e.$$.dirty[t/31|0]|=1<<t%31}function ae(e,t,n,r,l,i,s,g=[-1]){const c=L;N(e);const o=e.$$={fragment:null,ctx:[],props:i,update:I,not_equal:l,bound:W(),on_mount:[],on_destroy:[],on_disconnect:[],before_update:[],after_update:[],context:new Map(t.context||(c?c.$$.context:[])),callbacks:W(),dirty:g,skip_bound:!1,root:t.target||c.$$.root};s&&s(o.root);let b=!1;if(o.ctx=n?n(e,t.props||{},(a,q,...y)=>{const h=y.length?y[0]:q;return o.ctx&&l(o.ctx[a],o.ctx[a]=h)&&(!o.skip_bound&&o.bound[a]&&o.bound[a](h),b&&ue(e,a)),q}):[],o.update(),b=!0,O(o.before_update),o.fragment=r?r(o.ctx):!1,t.target){if(t.hydrate){const a=ne(t.target);o.fragment&&o.fragment.l(a),a.forEach(z)}else o.fragment&&o.fragment.c();t.intro&&se(e.$$.fragment),ce(e,t.target,t.anchor,t.customElement),H()}N(c)}class de{$destroy(){fe(this,1),this.$destroy=I}$on(t,n){if(!X(n))return I;const r=this.$$.callbacks[t]||(this.$$.callbacks[t]=[]);return r.push(n),()=>{const l=r.indexOf(n);l!==-1&&r.splice(l,1)}}$set(t){this.$$set&&!x(t)&&(this.$$.skip_bound=!0,this.$$set(t),this.$$.skip_bound=!1)}}const be="";function J(e,t,n){const r=e.slice();return r[11]=t[n],r[13]=n,r}function K(e){let t,n=e[11]+"",r,l,i,s;function g(){return e[9](e[13])}return{c(){t=k("button"),r=A(n),l=S(),v(t,"class","mlu-quizquestion-option-button button svelte-fk6ar3"),p(t,"active",e[5]===e[13]),p(t,"correct-answer",e[1]&&e[5]===e[13]&&e[2]==e[0].correctIndex),p(t,"incorrect-answer",e[1]&&e[5]===e[13]&&e[2]!=e[0].correctIndex),p(t,"available",e[4]&&e[1])},m(c,o){j(c,t,o),m(t,r),m(t,l),i||(s=Y(t,"click",g),i=!0)},p(c,o){e=c,o&1&&n!==(n=e[11]+"")&&T(r,n),o&32&&p(t,"active",e[5]===e[13]),o&39&&p(t,"correct-answer",e[1]&&e[5]===e[13]&&e[2]==e[0].correctIndex),o&39&&p(t,"incorrect-answer",e[1]&&e[5]===e[13]&&e[2]!=e[0].correctIndex),o&18&&p(t,"available",e[4]&&e[1])},d(c){c&&z(t),i=!1,s()}}}function U(e){let t;function n(i,s){return i[3]==!0?he:_e}let r=n(e),l=r(e);return{c(){l.c(),t=te()},m(i,s){l.m(i,s),j(i,t,s)},p(i,s){r!==(r=n(i))&&(l.d(1),l=r(i),l&&(l.c(),l.m(t.parentNode,t)))},d(i){l.d(i),i&&z(t)}}}function _e(e){let t;return{c(){t=k("p"),t.textContent="This is not the correct answer. Try again!",v(t,"class","svelte-fk6ar3")},m(n,r){j(n,t,r)},d(n){n&&z(t)}}}function he(e){let t;return{c(){t=k("p"),t.textContent="Good! You got the correct answer.",v(t,"class","svelte-fk6ar3")},m(n,r){j(n,t,r)},d(n){n&&z(t)}}}function me(e){let t,n,r=e[0].question+"",l,i,s,g,c,o,b=e[1]?"Retry":"Submit",a,q,y,h,C=e[0].options,d=[];for(let f=0;f<C.length;f+=1)d[f]=K(J(e,C,f));let _=e[1]&&U(e);return{c(){t=k("div"),n=k("h2"),l=A(r),i=S(),s=k("div");for(let f=0;f<d.length;f+=1)d[f].c();g=S(),c=k("div"),o=k("button"),a=A(b),q=S(),_&&_.c(),v(n,"class","svelte-fk6ar3"),v(s,"class","options svelte-fk6ar3"),v(o,"class","submit-button svelte-fk6ar3"),p(o,"available",!e[4]),v(c,"class","footer svelte-fk6ar3"),v(t,"class","quiz-wrapper svelte-fk6ar3")},m(f,w){j(f,t,w),m(t,n),m(n,l),m(t,i),m(t,s);for(let u=0;u<d.length;u+=1)d[u].m(s,null);m(t,g),m(t,c),m(c,o),m(o,a),m(c,q),_&&_.m(c,null),y||(h=Y(o,"click",e[10]),y=!0)},p(f,[w]){if(w&1&&r!==(r=f[0].question+"")&&T(l,r),w&311){C=f[0].options;let u;for(u=0;u<C.length;u+=1){const V=J(f,C,u);d[u]?d[u].p(V,w):(d[u]=K(V),d[u].c(),d[u].m(s,null))}for(;u<d.length;u+=1)d[u].d(1);d.length=C.length}w&2&&b!==(b=f[1]?"Retry":"Submit")&&T(a,b),w&16&&p(o,"available",!f[4]),f[1]?_?_.p(f,w):(_=U(f),_.c(),_.m(c,null)):_&&(_.d(1),_=null)},i:I,o:I,d(f){f&&z(t),ee(d,f),_&&_.d(),y=!1,h()}}}function pe(e,t,n){let{question:r={question:"Who didn't attend this meeting?",options:["Xin","Anand","Brent"],correctIndex:2}}=t,l=!1,i=-1,s="no",g=!1,c;function o(){n(4,g=!1),n(1,l=!1),n(2,i=-1),n(5,c=-1),n(3,s="no")}function b(){n(1,l=!0),n(3,s=i==r.correctIndex)}function a(h){n(4,g=!0),n(2,i=h),n(5,c=h)}const q=h=>a(h),y=()=>l?o():b();return e.$$set=h=>{"question"in h&&n(0,r=h.question)},[r,l,i,s,g,c,o,b,a,q,y]}class ge extends de{constructor(t){super(),ae(this,t,pe,me,Z,{question:0})}}return ge}();
</script>"""
        self.div_id = Quiz.randomize_hash_id(self.name)
        self.markup = ""
        self.add_params(kwargs)

    def add_params(self, params: Dict[str, Any]) -> None:
        """
        Add parameters to the component and serve in html.

        Parameters
        ----------
        params : dict
            The parameters to add to the component.
        """
        js_data = json.dumps(params, indent=0)
        self.markup = f"""
        <div id="{self.div_id}"></div>
        <script>
        (() => {{
            var data = {js_data};
            window.{self.name}_data = data;
            var {self.name}_inst = new {self.name}({{
                "target": document.getElementById("{self.div_id}"),
                "props": data
            }});
        }})();
        </script>
        """

    def _repr_html_(self) -> str:
        """
        Return the component as an HTML string.
        """
        return f"""
        {self.iife_script}
        {self.markup}
        """

    @staticmethod
    def randomize_hash_id(name) -> str:
        """
        Generate a random hash for div id for a component.

        Returns
        -------
        str
            The generated hash ID.
        """
        random_hex_id = hex(randint(10**8, 10**9))[2:]
        hashed_div_id = f"{name}-{random_hex_id}"
        return hashed_div_id


lab2a_question1 = Quiz(
    question={
        "question": "Which model parameter controls randomness or diversity in generated sequences?", "options" : [
            "temperature",
            "topP",
            "MaxTokens",
            "stopSequences",
        ],
        "correctIndex": 0,
    }
)


lab2b_question1 = Quiz(
    question={
        "question": "Which module was used for memory in the conversational chain?", "options" : [
            "PromptTemplate",
            "BedrockLLM",
            "ConversationBufferMemory",
            "ConversationChain",
        ],
        "correctIndex": 2,
    }
)

lab3_question1 = Quiz(
    question={
        "question": "Which of the following is NOT a terminology used for of in-context learning techniques?", "options" : [
            "Few-shot learning",
            "Zero-shot learning",
            "One-shot learning",
            "Some-shot learning",
        ],
        "correctIndex": 3,
    }
)

lab3_question2 = Quiz(
    question={
        "question": "Chain-of-Thought (CoT) is most preferred for which of the following tasks?", "options" : [
            "Arithmetic Reasoning",
            "Summarization of essays",
            "Factual Q&A",
            "Machine translation",
        ],
        "correctIndex": 0,
    }
)

lab4a_question1 = Quiz(
    question={
        "question": "Which of the following outcomes is TRUE when temperature is set to zero?", "options" : [
            "You can use self-consistency technique.",
            "You can use the Tree-of-thought technique.",
            "You can use Chain-of-Thought technique.",
            "None of the above.",
        ],
        "correctIndex": 2,
    }
)

lab4b_question1 = Quiz(
    question={
        "question": "Which of the following techniques is expected to have the highest prompting cost?", "options" : [
            "Chain-of-thought (CoT)",
            "Self-consistency",
            "Tree-of-thought (ToT)",
            "Zro-shot prompting",
        ],
        "correctIndex": 2,
    }
)


lab5_question1 = Quiz(
    question={
        "question": "Which of the following is NOT a supported image format for Claude3 models?", "options" : [
            ".png",
            ".mp4",
            ".gif",
            ".jpg",
        ],
        "correctIndex": 2,
    }
)

lab5_question2 = Quiz(
    question={
        "question": "Which statement is true when natively prompting Claude3 models with GIFs.", "options":[
            "Only the first frame is considered",
            "GIFs are not supported by Claude3.",
            "The entire sequence of frames are natively considered",
            "Only the last frame is considered.",
        ],
        "correctIndex": 0,
    }
)